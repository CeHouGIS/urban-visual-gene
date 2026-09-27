from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/graph_visual_vocabulary.yaml"


def assert_safe_affinity() -> None:
    if hasattr(os, "sched_getaffinity"):
        unsafe = set(os.sched_getaffinity(0)).intersection({8, 9})
        if unsafe:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(unsafe)}")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    for key, raw in value["paths"].items():
        value["paths"][key] = Path(raw)
    value["semantic"]["model_dir"] = Path(value["semantic"]["model_dir"])
    return value


def ensure_layout(config: dict[str, Any]) -> None:
    cache = config["paths"]["cache_root"]
    data = config["paths"]["paper_data_root"]
    figures = config["paths"]["paper_figure_root"]
    for path in (
        cache / "views", cache / "graph", cache / "leiden/partitions",
        cache / "validation", cache / "downstream", cache / "semantic",
        data / "graph", data / "leiden", data / "validation",
        data / "downstream/city_ppmi", data / "downstream/spatial_metrics",
        data / "semantic/raw_outputs", figures / "community_evidence",
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def l2_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def cosine_rows(values: np.ndarray) -> np.ndarray:
    normalized = l2_rows(values)
    return np.clip(normalized @ normalized.T, -1, 1)


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    groups = sorted(np.unique(labels), key=lambda value: int(np.flatnonzero(labels == value)[0]))
    mapping = {value: index for index, value in enumerate(groups)}
    return np.asarray([mapping[value] for value in labels], dtype=np.int16)


def gamma_values(config: dict[str, Any]) -> np.ndarray:
    section = config["leiden"]
    values = np.arange(
        float(section["gamma_min"]),
        float(section["gamma_max"]) + float(section["gamma_step"]) / 2,
        float(section["gamma_step"]),
    )
    return np.round(values, 10)


def should_skip(paths: list[Path], force: bool) -> bool:
    return not force and all(path.exists() for path in paths)
