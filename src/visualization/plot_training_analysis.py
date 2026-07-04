#!/usr/bin/env python3
"""
Generates 5 analysis plots summarizing the RL training experiments documented in
EXPERIMENTS.md, saved to visualizations/analysis/:

    1. Reward learning curves (update-frequency and scratch-vs-fine-tuned comparisons)
    2. Policy entropy + VTR cache hit rate over training (scratch vs. fine-tuned)
    3. Area x Delay x Power (ADP) summary bar chart across every tested condition
    4. Area-vs-delay scatter, colored by episode index, per run
    5. DSP placement frequency heatmap, per run

Usage:
    python3 src/visualization/plot_training_analysis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "visualizations" / "analysis"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
})

RUN_FILES = {
    "diffeq2_120update": PROJECT_ROOT / "all_layouts_diffeq2_seed_42_fullscale.jsonl",
    "diffeq2_24update":  PROJECT_ROOT / "all_layouts_diffeq2_seed_42_24perupdate.jsonl",
    "diffeq1_scratch":   PROJECT_ROOT / "all_layouts_diffeq1_seed_42_scratch.jsonl",
    "diffeq1_finetuned": PROJECT_ROOT / "all_layouts_diffeq1_seed_42_finetuned.jsonl",
}

TB_LOGDIRS = {
    "diffeq1_scratch":   PROJECT_ROOT / "runs" / "tb_logs" / "MaskablePPO_5",
    "diffeq1_finetuned": PROJECT_ROOT / "runs" / "tb_logs" / "MaskablePPO_6",
}

GRID_W, GRID_H = 14, 14

# Final best-layout metrics for every tested condition, sourced from the
# corresponding best_layout_coordinates_*.txt / zero_shot_transfer_*.json files
# generated during the experiments in EXPERIMENTS.md.
ADP_DATA = [
    ("diffeq2", "Traditional",              666210.0, 17.948,  0.007747),
    ("diffeq2", "RL 120/update (6000 ep)",  298474.0, 17.2762, 0.005506),
    ("diffeq2", "RL 24/update (2016 ep)",   324551.0, 17.6789, 0.005574),
    ("diffeq1", "Traditional",              694168.0, 22.3395, 0.007714),
    ("diffeq1", "Random (best of 20)",      575810.0, 22.1378, 0.006892),
    ("diffeq1", "RL zero-shot (0 ep)",      449975.0, 21.5744, 0.006178),
    ("diffeq1", "RL from scratch (6000 ep)", 402316.0, 21.6266, 0.005868),
    ("diffeq1", "RL fine-tuned (6000 ep)",  414992.0, 21.1079, 0.006042),
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_run(name: str) -> pd.DataFrame:
    records = []
    with open(RUN_FILES[name]) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)
    df["episode"] = np.arange(1, len(df) + 1)
    return df


def mark_cache_hits(df: pd.DataFrame) -> pd.DataFrame:
    """A cache hit is any (dsps, brams, aspect_ratio) combo already seen earlier
    in the same run — mirrors LayoutCache's key exactly (src/env/fpga_env.py)."""
    seen = set()
    hits = []
    for dsps, brams, ratio in zip(df["dsps"], df["brams"], df["aspect_ratio"]):
        key = (tuple(map(tuple, dsps)), tuple(map(tuple, brams)), ratio)
        hits.append(key in seen)
        seen.add(key)
    df = df.copy()
    df["cache_hit"] = hits
    return df


def load_tb_scalar(logdir: Path, tag: str) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    events = ea.Scalars(tag)
    return pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})


# ── Plot 1: reward learning curves ──────────────────────────────────────────────

def plot_reward_curves():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    panels = [
        (axes[0], "diffeq2 — update frequency",
         [("diffeq2_120update", "120 episodes/update", "#1d4ed8"),
          ("diffeq2_24update",  "24 episodes/update",  "#b91c1c")]),
        (axes[1], "diffeq1 — scratch vs. fine-tuned",
         [("diffeq1_scratch",   "From scratch",            "#1d4ed8"),
          ("diffeq1_finetuned", "Fine-tuned from diffeq2", "#b91c1c")]),
    ]

    for ax, title, series in panels:
        for name, label, color in series:
            df = load_run(name)
            roll = df["reward"].rolling(100, min_periods=20)
            mean, std = roll.mean(), roll.std()
            ax.plot(df["episode"], mean, label=label, color=color, linewidth=1.8)
            ax.fill_between(df["episode"], mean - std, mean + std, color=color, alpha=0.15)
        ax.axhline(0, color="#94a3b8", linewidth=0.9, linestyle="--", label="Baseline parity")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (100-episode rolling mean ± std)")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Reward learning curves", fontsize=15)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "1_reward_learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 1_reward_learning_curves.png")


# ── Plot 2: entropy + cache hit rate ────────────────────────────────────────────

def plot_entropy_and_cache():
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    colors = {"diffeq1_scratch": "#1d4ed8", "diffeq1_finetuned": "#b91c1c"}
    labels = {"diffeq1_scratch": "From scratch", "diffeq1_finetuned": "Fine-tuned from diffeq2"}

    ax = axes[0]
    for name in ("diffeq1_scratch", "diffeq1_finetuned"):
        tb = load_tb_scalar(TB_LOGDIRS[name], "train/policy_entropy")
        episodes = tb["step"] / 6.0  # 6 env-steps per diffeq1 episode (1 aspect-ratio + 5 DSPs)
        ax.plot(episodes, tb["value"], label=labels[name], color=colors[name], linewidth=1.8)
    ax.set_title("Policy entropy over training")
    ax.set_ylabel("Policy entropy")
    ax.set_xlabel("Episode (approx.)")
    ax.legend()

    ax = axes[1]
    for name in ("diffeq1_scratch", "diffeq1_finetuned"):
        df = mark_cache_hits(load_run(name))
        rolling_hit_rate = df["cache_hit"].rolling(200, min_periods=20).mean()
        ax.plot(df["episode"], rolling_hit_rate, label=labels[name], color=colors[name], linewidth=1.8)
    ax.set_title("VTR cache hit rate (200-episode rolling window)")
    ax.set_ylabel("Fraction of episodes resolved by cache hit")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.suptitle("Entropy collapse drives the cache-hit speedup (fine-tuned vs. scratch)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "2_entropy_and_cache_hits.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 2_entropy_and_cache_hits.png")


# ── Plot 3: ADP summary bar chart ───────────────────────────────────────────────

def plot_adp_summary():
    df = pd.DataFrame(ADP_DATA, columns=["benchmark", "condition", "area", "delay", "power"])
    df["adp"] = df["area"] * df["delay"] * df["power"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, bm in zip(axes, ("diffeq2", "diffeq1")):
        sub = df[df["benchmark"] == bm].sort_values("adp", ascending=False).reset_index(drop=True)
        base = sub["adp"].iloc[0]  # Traditional is always the worst (highest) ADP
        colors = ["#94a3b8" if c == "Traditional" else "#b91c1c" if "Random" in c else "#1d4ed8"
                  for c in sub["condition"]]
        bars = ax.barh(sub["condition"], sub["adp"], color=colors)
        for bar, val in zip(bars, sub["adp"]):
            pct = (base - val) / base * 100
            label = f"{val:,.0f}" + (f"   (-{pct:.1f}%)" if pct > 0.5 else "")
            ax.text(val + base * 0.01, bar.get_y() + bar.get_height() / 2, label,
                     va="center", fontsize=9, color="#1f2328")
        ax.set_title(f"{bm} — Area × Delay × Power")
        ax.set_xlabel("ADP (lower is better)")
        ax.invert_yaxis()
        ax.set_xlim(0, base * 1.35)

    fig.suptitle("ADP comparison across every tested condition", fontsize=15)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "3_adp_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 3_adp_summary.png")


# ── Plot 4: area-vs-delay scatter colored by episode ────────────────────────────

def plot_metric_scatter():
    names = ["diffeq2_120update", "diffeq2_24update", "diffeq1_scratch", "diffeq1_finetuned"]
    titles = ["diffeq2 — 120/update (6000 ep)", "diffeq2 — 24/update (2016 ep)",
              "diffeq1 — from scratch (6000 ep)", "diffeq1 — fine-tuned (6000 ep)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, name, title in zip(axes.flat, names, titles):
        df = load_run(name)
        df = df[df["success"] == 1]
        sc = ax.scatter(df["routing_area"], df["delay_ns"], c=df["episode"],
                         cmap="viridis", s=14, alpha=0.6, edgecolors="none")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Routing area (log scale)")
        ax.set_ylabel("Delay (ns)")
        fig.colorbar(sc, ax=ax, label="Episode")

    fig.suptitle("Explored area-delay tradeoff space over training", fontsize=15)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "4_area_delay_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 4_area_delay_scatter.png")


# ── Plot 5: DSP placement heatmap ───────────────────────────────────────────────

DSP_HEIGHT = 4  # a DSP placed at (x, y) occupies rows y .. y+DSP_HEIGHT-1 (src/env/fpga_env.py)


def plot_dsp_heatmap():
    names = ["diffeq2_120update", "diffeq2_24update", "diffeq1_scratch", "diffeq1_finetuned"]
    titles = ["diffeq2 — 120/update", "diffeq2 — 24/update",
              "diffeq1 — from scratch", "diffeq1 — fine-tuned"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, name, title in zip(axes.flat, names, titles):
        df = load_run(name)
        df = df[df["success"] == 1]
        heat = np.zeros((GRID_H, GRID_W))
        for dsps in df["dsps"]:
            for x, y in dsps:
                if 1 <= x <= GRID_W and 1 <= y <= GRID_H:
                    for dy in range(DSP_HEIGHT):
                        if y - 1 + dy < GRID_H:
                            heat[y - 1 + dy, x - 1] += 1
        sns.heatmap(heat, ax=ax, cmap="rocket_r", cbar_kws={"label": "Tile covered by a DSP (count)"})
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()

    fig.suptitle("DSP tile coverage frequency — full 4-row footprint per placement (all evaluated episodes)",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "5_dsp_placement_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 5_dsp_placement_heatmap.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_reward_curves()
    plot_entropy_and_cache()
    plot_adp_summary()
    plot_metric_scatter()
    plot_dsp_heatmap()


if __name__ == "__main__":
    main()
