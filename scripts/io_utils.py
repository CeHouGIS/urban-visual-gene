"""Shared I/O helpers and checkpoint assertions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def checkpoint(condition: bool, msg: str) -> None:
    if not condition:
        print(f"[CHECKPOINT FAIL] {msg}", file=sys.stderr)
        sys.exit(1)


def checkpoint_warn(condition: bool, msg: str) -> None:
    if not condition:
        print(f"[CHECKPOINT WARN] {msg}", file=sys.stderr)


def save_report(path: Path | str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def assert_l2_normed(emb_array: np.ndarray, name: str = "embeddings",
                     atol: float = 1e-5) -> None:
    norms = np.linalg.norm(emb_array, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError(
            f"[CHECKPOINT FAIL] {name} not L2-normed: "
            f"max deviation = {np.abs(norms - 1).max():.6f}"
        )
