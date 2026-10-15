"""Shared experiment I/O and indexed street-view metadata lookup."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def save_torch(path, value):
    import torch

    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def metadata_for(panoids: set[str], metadata_csv: Path) -> dict[str, tuple[float, float]]:
    import pandas as pd

    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    found: dict[str, tuple[float, float]] = {}
    for chunk in pd.read_csv(metadata_csv, usecols=["panoid", "lat", "lon"],
                             dtype={"panoid": str}, chunksize=200_000):
        hits = chunk[chunk.panoid.isin(panoids) & chunk.lat.notna() & chunk.lon.notna()]
        for row in hits.itertuples(index=False):
            found[row.panoid] = (float(row.lat), float(row.lon))
        if len(found) == len(panoids):
            break
    return found
