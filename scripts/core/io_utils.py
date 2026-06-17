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


def stack_embeddings(values, D: int | None = None) -> np.ndarray:
    """Stack a sequence of embedding arrays into a clean (N, D) float32 matrix.

    Robust to rare wrong-length rows: the environment's intermittent native
    numpy/scipy corruption occasionally writes a malformed embedding (e.g. a
    3081- or 3264-long row among 3072-long ones), which would break np.stack.
    The target D defaults to the most common row length; any row whose length
    differs is replaced with a zero vector (and the count is reported).
    """
    arrs = [np.asarray(v, dtype=np.float32).ravel() for v in values]
    if D is None:
        from collections import Counter
        D = Counter(a.shape[0] for a in arrs).most_common(1)[0][0]
    out = np.zeros((len(arrs), D), dtype=np.float32)
    n_bad = 0
    for i, a in enumerate(arrs):
        if a.shape[0] == D:
            out[i] = a
        else:
            n_bad += 1
    if n_bad:
        print(f"[stack_embeddings] coerced {n_bad}/{len(arrs)} malformed "
              f"rows (len≠{D}) to zero vectors", file=sys.stderr)
    return out


def assert_l2_normed(emb_array: np.ndarray, name: str = "embeddings",
                     atol: float = 1e-5) -> None:
    norms = np.linalg.norm(emb_array, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError(
            f"[CHECKPOINT FAIL] {name} not L2-normed: "
            f"max deviation = {np.abs(norms - 1).max():.6f}"
        )
