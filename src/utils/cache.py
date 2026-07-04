#!/usr/bin/env python3

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheRow:
    delay_ns: float
    wirelength: float
    power_w: float
    routing_area: float
    grid_w: int
    grid_h: int
    success: bool


_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS layout_cache (
        cache_key   TEXT PRIMARY KEY,
        delay_ns    REAL,
        wirelength  REAL,
        power_w     REAL,
        routing_area REAL DEFAULT 0,
        grid_w      REAL DEFAULT 0,
        grid_h      REAL DEFAULT 0,
        success     INTEGER
    )
"""

_MIGRATION_COLS = ["grid_w", "grid_h", "routing_area"]


class LayoutCache:
    """Thread-safe SQLite cache for VTR layout evaluation results."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=60.0)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_SQL)
            for col in _MIGRATION_COLS:
                try:
                    conn.execute(f"ALTER TABLE layout_cache ADD COLUMN {col} REAL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass

    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[CacheRow]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT delay_ns, wirelength, power_w, success, grid_w, grid_h, routing_area"
                    " FROM layout_cache WHERE cache_key=?",
                    (key,),
                ).fetchone()
            if row is None:
                return None
            delay_ns, wirelength, power_w, success, grid_w, grid_h, routing_area = row
            return CacheRow(
                delay_ns=delay_ns,
                wirelength=wirelength,
                power_w=power_w,
                routing_area=routing_area or 0.0,
                grid_w=int(grid_w or 0),
                grid_h=int(grid_h or 0),
                success=bool(success),
            )
        except Exception:
            return None

    def put(self, key: str, row: CacheRow) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO layout_cache VALUES (?,?,?,?,?,?,?,?)",
                    (
                        key,
                        row.delay_ns,
                        row.wirelength,
                        row.power_w,
                        row.routing_area,
                        row.grid_w,
                        row.grid_h,
                        row.success,
                    ),
                )
        except Exception:
            pass

    @staticmethod
    def failure_row() -> CacheRow:
        return CacheRow(
            delay_ns=float("inf"),
            wirelength=float("inf"),
            power_w=float("inf"),
            routing_area=float("inf"),
            grid_w=0,
            grid_h=0,
            success=False,
        )
