#!/usr/bin/env python3

import sys

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
import wandb
from sb3_contrib import MaskablePPO


class CustomMaskablePPO(MaskablePPO):
    """
    MaskablePPO extended with per-environment gradient variance diagnostics.

    After each rollout buffer is collected, we compute the following 7 stability
    metrics before delegating to the standard PPO update:

        1. batch_reward_variance   — variance of returns at step 0 across envs
        2. gradient_variance_norm  — L2 norm of per-parameter gradient variance
        3. avg_cosine_similarity   — mean cosine similarity between per-env
                                     gradients and the mean gradient
        4. global_grad_norm        — L2 norm of the mean gradient vector
        5. policy_entropy          — mean entropy of the policy distribution
        6. value_loss              — mean MSE between returns and value estimates
        7. explained_variance      — 1 - Var(returns - values) / Var(returns)
    """

    def train(self) -> None:
        try:
            self._compute_stability_metrics()
        except Exception as exc:
            print(f"[Warning] Stability metrics failed: {exc}", file=sys.stderr)
        super().train()

    # ------------------------------------------------------------------

    def _compute_stability_metrics(self) -> None:
        self.policy.set_training_mode(True)
        params = [p for p in self.policy.parameters() if p.requires_grad]

        flat_grads: list[th.Tensor] = []
        all_entropies: list[th.Tensor] = []
        all_vf_losses: list[th.Tensor] = []

        clip_range_val = self.clip_range(self._current_progress_remaining)

        for env_idx in range(self.n_envs):
            if isinstance(self.rollout_buffer.observations, dict):
                obs_i = {
                    k: th.as_tensor(v[:, env_idx]).to(self.device)
                    for k, v in self.rollout_buffer.observations.items()
                }
            else:
                obs_i = th.as_tensor(self.rollout_buffer.observations[:, env_idx]).to(self.device)
            actions_i = th.as_tensor(self.rollout_buffer.actions[:, env_idx]).to(self.device)
            advantages_i = th.as_tensor(self.rollout_buffer.advantages[:, env_idx]).to(self.device)
            returns_i = th.as_tensor(self.rollout_buffer.returns[:, env_idx]).to(self.device)
            old_log_prob_i = th.as_tensor(self.rollout_buffer.log_probs[:, env_idx]).to(self.device)

            action_masks_i = None
            if hasattr(self.rollout_buffer, "action_masks") and self.rollout_buffer.action_masks is not None:
                action_masks_i = th.as_tensor(self.rollout_buffer.action_masks[:, env_idx]).to(self.device)

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions_i = actions_i.long().flatten()

            values_i, log_prob_i, entropy_i = self.policy.evaluate_actions(
                obs_i, actions_i, action_masks=action_masks_i
            )
            values_i = values_i.flatten()

            if self.normalize_advantage:
                advantages_i = (advantages_i - advantages_i.mean()) / (advantages_i.std() + 1e-8)

            ratio_i = th.exp(log_prob_i - old_log_prob_i)
            policy_loss_i = -th.min(
                advantages_i * ratio_i,
                advantages_i * th.clamp(ratio_i, 1 - clip_range_val, 1 + clip_range_val),
            ).mean()

            if self.clip_range_vf is not None:
                clip_vf = self.clip_range_vf(self._current_progress_remaining)
                old_values_i = th.as_tensor(self.rollout_buffer.values[:, env_idx]).to(self.device)
                values_pred_i = old_values_i + th.clamp(values_i - old_values_i, -clip_vf, clip_vf)
            else:
                values_pred_i = values_i

            vf_loss_i = F.mse_loss(returns_i, values_pred_i)
            all_vf_losses.append(vf_loss_i)

            if entropy_i is None:
                entropy_loss_i = -th.mean(-log_prob_i)
                all_entropies.append(-log_prob_i.mean())
            else:
                entropy_loss_i = -th.mean(entropy_i)
                all_entropies.append(entropy_i.mean())

            loss_i = policy_loss_i + self.ent_coef * entropy_loss_i + self.vf_coef * vf_loss_i
            self.policy.optimizer.zero_grad()
            loss_i.backward()

            flat_grads.append(
                th.cat([p.grad.view(-1) if p.grad is not None else th.zeros(p.numel(), device=self.device) for p in params])
            )

        g = th.stack(flat_grads)              # (n_envs, num_params)
        g_bar = g.mean(dim=0)

        global_grad_norm = th.norm(g_bar).item()
        gradient_variance_norm = th.norm(th.var(g, dim=0, unbiased=False)).item()

        g_norms = th.norm(g, dim=1, keepdim=True) + 1e-8
        cos_sims = th.sum(g * g_bar, dim=1, keepdim=True) / (g_norms * (th.norm(g_bar) + 1e-8))
        avg_cosine_similarity = cos_sims.mean().item()

        batch_reward_variance = float(np.var(self.rollout_buffer.returns[0]))
        policy_entropy = th.mean(th.stack(all_entropies)).item()
        vf_loss = th.mean(th.stack(all_vf_losses)).item()

        y_true = self.rollout_buffer.returns.flatten()
        y_pred = self.rollout_buffer.values.flatten()
        var_y = np.var(y_true)
        explained_var = float(1.0 - np.var(y_true - y_pred) / var_y if var_y > 1e-8 else 0.0)

        self.policy.optimizer.zero_grad()

        self.latest_metrics = {
            "batch_reward_variance": batch_reward_variance,
            "gradient_variance_norm": gradient_variance_norm,
            "avg_cosine_similarity": avg_cosine_similarity,
            "global_grad_norm": global_grad_norm,
            "policy_entropy": policy_entropy,
            "value_loss": vf_loss,
            "explained_variance": explained_var,
        }

        if wandb.run is not None:
            wandb.log(
                {f"train/{k}": v for k, v in self.latest_metrics.items()},
                step=self.num_timesteps,
            )

        print(f"\n[Stability Report — update #{self._n_updates + 1}]")
        for k, v in self.latest_metrics.items():
            print(f"  {k:<35}: {v:.6f}")
