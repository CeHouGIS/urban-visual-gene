#!/usr/bin/env python3
"""Filter the 30-city image manifests before feature-MAE training.

The retained legacy ``quality_model.joblib`` detects over-exposure/glare,
dark/corrupt and other low-information frames.  The project's calibrated
black/blur/tunnel rules are applied as a second, auditable guard.  Images stay
inside the source TARs; this script only writes clean/drop manifests.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from scipy import ndimage

from scripts.multicity.config import CITIES, city_slug
from scripts.multicity.patch_config import OUTPUT_ROOT as SOURCE_ROOT
from scripts.quality.image_quality import FEAT_COLS, image_features

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_OUTPUT = SOURCE_ROOT.parent / "feature_mae_n30x12800_qc"
DEFAULT_MODEL = Path("models/quality_model.joblib")
CONDA_PYTHON = Path("/opt/conda/bin/python")
RESOLUTION = 256
GRID = 4
TILE = RESOLUTION // GRID
TUNNEL_THRESHOLDS = {
    "black_mean": 14.0,
    "dark_frac": 0.97,
    "std_min": 6.0,
    "blur_tilefrac": 0.4,
    "blur_lapvar": 48.0,
    "tunnel_bright": 100.0,
    "tunnel_sky": 95.0,
    "tunnel_lapvar": 280.0,
    "tunnel_warm": 4.0,
    "tunnel_glow": 42.0,
}

_FDS: OrderedDict[str, int] = OrderedDict()


def _fd(path: str, max_open: int = 32) -> int:
    if path in _FDS:
        handle = _FDS.pop(path)
        _FDS[path] = handle
        return handle
    handle = os.open(path, os.O_RDONLY)
    _FDS[path] = handle
    if len(_FDS) > max_open:
        _, old = _FDS.popitem(last=False)
        os.close(old)
    return handle


def _formal_metrics(rgb255: np.ndarray) -> tuple[float, ...]:
    luminance = rgb255 @ np.array([0.299, 0.587, 0.114], np.float32)
    bright = float(luminance.mean())
    dark = float((luminance < 20).mean())
    std = float(luminance.std())
    lap = ndimage.laplace(luminance)
    lapvar = float(lap.var())
    tiles = (
        luminance.reshape(GRID, TILE, GRID, TILE)
        .transpose(0, 2, 1, 3)
        .reshape(GRID * GRID, -1)
    )
    lap_tiles = (
        lap.reshape(GRID, TILE, GRID, TILE)
        .transpose(0, 2, 1, 3)
        .reshape(GRID * GRID, -1)
    )
    blur_tilefrac = float(((lap_tiles.var(1) < 12) & (tiles.std(1) > 8)).mean())
    warm = float(rgb255[..., 0].mean() - rgb255[..., 2].mean())
    skytop = float(luminance[: RESOLUTION // 4].mean())
    high = np.percentile(luminance, 95)
    glow = float(luminance[luminance >= high].mean() - bright)
    return bright, dark, std, lapvar, blur_tilefrac, warm, skytop, glow


def _rule_flags(values: tuple[float, ...]) -> tuple[bool, bool, bool]:
    bright, dark, std, lapvar, blur_tilefrac, warm, skytop, glow = values
    t = TUNNEL_THRESHOLDS
    black = (
        bright < t["black_mean"] or dark > t["dark_frac"]
        or std < t["std_min"] or bright < 0
    )
    blur = blur_tilefrac > t["blur_tilefrac"] or 0 <= lapvar < t["blur_lapvar"]
    tunnel = (
        not black
        and bright < t["tunnel_bright"]
        and skytop < t["tunnel_sky"]
        and lapvar < t["tunnel_lapvar"]
        and (warm > t["tunnel_warm"] or glow > t["tunnel_glow"])
    )
    return bool(black), bool(blur), bool(tunnel)


def _analyze(row: tuple[int, str, int, int]) -> tuple:
    image_index, tar_path, offset, size = row
    try:
        payload = os.pread(_fd(tar_path), size, offset)
        if len(payload) != size:
            raise OSError(f"short read: {len(payload)} != {size}")
        with Image.open(io.BytesIO(payload)) as image:
            rgb255 = np.asarray(
                image.convert("RGB").resize((RESOLUTION, RESOLUTION)),
                dtype=np.float32,
            )
        learned = image_features(rgb255 / 255.0)
        formal = _formal_metrics(rgb255)
        black, blur, tunnel = _rule_flags(formal)
        return (
            image_index, False, *[learned[name] for name in FEAT_COLS],
            *formal, black, blur, tunnel,
        )
    except Exception:
        return (
            image_index, True, *([0.0] * len(FEAT_COLS)),
            *([-1.0] * 8), True, False, False,
        )


def _predict_joblib(features: Path, output: Path, model_path: Path) -> None:
    import joblib

    bundle = joblib.load(model_path)
    model = bundle["model"]
    columns = list(bundle["features"])
    if columns != list(FEAT_COLS):
        raise ValueError(f"quality-model feature mismatch: {columns} != {FEAT_COLS}")
    values = np.load(features, mmap_mode="r")
    prediction = model.predict_proba(values)[:, 1].astype(np.float32)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, prediction)
    os.replace(temporary, output)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def filter_city(
    city: str,
    source_root: Path,
    output_root: Path,
    model_path: Path,
    threshold: float,
    workers: int,
) -> dict:
    import pandas as pd

    slug = city_slug(city)
    source_path = source_root / "manifests" / f"{slug}.parquet"
    clean_path = output_root / "filtered_manifests" / f"{slug}.parquet"
    dropped_path = output_root / "dropped_manifests" / f"{slug}.parquet"
    qc_path = output_root / "qc" / f"{slug}.parquet"
    manifest = pd.read_parquet(source_path).sort_values("image_index").reset_index(drop=True)

    if clean_path.exists() and dropped_path.exists() and qc_path.exists():
        clean = pd.read_parquet(clean_path, columns=["source_image_index"])
        dropped = pd.read_parquet(dropped_path, columns=["source_image_index", "qc_reason"])
        if len(clean) + len(dropped) == len(manifest):
            reasons = dropped["qc_reason"].str.get_dummies(sep="|").sum().to_dict()
            return {
                "city_key": city,
                "source_images": len(manifest),
                "kept_images": len(clean),
                "dropped_images": len(dropped),
                "drop_rate": len(dropped) / len(manifest),
                "reason_counts": {str(k): int(v) for k, v in reasons.items()},
                "status": "existing",
            }

    for directory in (clean_path.parent, dropped_path.parent, qc_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = list(
        manifest[["image_index", "tar_path", "jpg_offset", "jpg_size"]]
        .itertuples(index=False, name=None)
    )
    started = time.time()
    rows: list[tuple] = []
    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap(_analyze, tasks, chunksize=32), 1):
            rows.append(result)
            if i % 2000 == 0 or i == len(tasks):
                print(
                    f"  {city}: QC {i:,}/{len(tasks):,} "
                    f"({i / max(time.time() - started, 1e-6):.1f} img/s)",
                    flush=True,
                )

    learned_columns = list(FEAT_COLS)
    formal_columns = [
        "bright", "dark_fraction", "std", "lapvar", "blur_tilefrac",
        "warm", "skytop", "glow",
    ]
    qc = pd.DataFrame(
        rows,
        columns=["source_image_index", "unreadable", *learned_columns,
                 *formal_columns, "black", "blur", "tunnel"],
    ).sort_values("source_image_index").reset_index(drop=True)
    feature_file = output_root / "qc" / f".{slug}.features.npy"
    probability_file = output_root / "qc" / f".{slug}.bad_prob.npy"
    np.save(feature_file, qc[learned_columns].to_numpy(np.float64))
    python = CONDA_PYTHON if CONDA_PYTHON.exists() else Path(sys.executable)
    subprocess.run(
        [str(python), "-m", "scripts.multicity.filter_image_quality", "--predict-npy",
         str(feature_file), "--prediction-output", str(probability_file),
         "--model", str(model_path.resolve())],
        check=True,
    )
    qc["artifact_bad_probability"] = np.load(probability_file)
    qc["model_bad"] = qc["artifact_bad_probability"] > threshold
    qc["is_bad"] = qc[["unreadable", "model_bad", "black", "blur", "tunnel"]].any(axis=1)

    def reason(row) -> str:
        labels = []
        for column, label in (
            ("unreadable", "unreadable"), ("model_bad", "artifact_model"),
            ("black", "black"), ("blur", "blur"), ("tunnel", "tunnel"),
        ):
            if bool(row[column]):
                labels.append(label)
        return "|".join(labels)

    qc["qc_reason"] = qc.apply(reason, axis=1)
    qc.to_parquet(qc_path, index=False, compression="zstd")
    enriched = manifest.copy()
    enriched["source_image_index"] = enriched["image_index"].astype(np.int32)
    enriched = enriched.merge(
        qc[["source_image_index", "artifact_bad_probability", "black", "blur",
            "tunnel", "unreadable", "is_bad", "qc_reason"]],
        on="source_image_index", how="left", validate="one_to_one",
    )
    clean = enriched[~enriched["is_bad"]].copy().reset_index(drop=True)
    clean["image_index"] = np.arange(len(clean), dtype=np.int32)
    dropped = enriched[enriched["is_bad"]].copy().reset_index(drop=True)
    clean.to_parquet(clean_path, index=False, compression="zstd")
    dropped.to_parquet(dropped_path, index=False, compression="zstd")
    feature_file.unlink(missing_ok=True)
    probability_file.unlink(missing_ok=True)
    reasons = dropped["qc_reason"].str.get_dummies(sep="|").sum().to_dict()
    return {
        "city_key": city,
        "source_images": len(manifest),
        "kept_images": len(clean),
        "dropped_images": len(dropped),
        "drop_rate": len(dropped) / len(manifest),
        "reason_counts": {str(k): int(v) for k, v in reasons.items()},
        "elapsed_seconds": time.time() - started,
        "status": "created",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", default=list(CITIES))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--predict-npy", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()
    if args.predict_npy:
        if args.prediction_output is None:
            parser.error("--prediction-output is required with --predict-npy")
        _predict_joblib(args.predict_npy, args.prediction_output, args.model)
        return
    if not args.model.exists():
        raise FileNotFoundError(f"quality model not found: {args.model}")
    reports = []
    for index, city in enumerate(args.cities, 1):
        print(f"[{index:02d}/{len(args.cities)}] filtering {city}", flush=True)
        report = filter_city(
            city, args.source_root, args.output_root, args.model,
            args.threshold, args.workers,
        )
        reports.append(report)
        print(
            f"  keep={report['kept_images']:,} drop={report['dropped_images']:,} "
            f"({report['drop_rate']:.1%})",
            flush=True,
        )
    summary = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "quality_model": str(args.model.resolve()),
        "model_threshold": args.threshold,
        "tunnel_filter": "calibrated legacy black/blur/tunnel rules",
        "cities": reports,
        "source_images": sum(x["source_images"] for x in reports),
        "kept_images": sum(x["kept_images"] for x in reports),
        "dropped_images": sum(x["dropped_images"] for x in reports),
    }
    summary["drop_rate"] = summary["dropped_images"] / summary["source_images"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_root / "qc_summary.json", summary)
    print(
        f"done: keep={summary['kept_images']:,}, drop={summary['dropped_images']:,} "
        f"({summary['drop_rate']:.1%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
