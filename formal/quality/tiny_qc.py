"""Low-cost panorama quality gate for the formal DINOv3 pipeline.

The gate is deliberately CPU-only and runs on 96px images.  A small
HistGradientBoosting model can be trained from weak labels produced by the
auditable rules below; inference still keeps the hard black-frame rule so a
missing or stale model cannot let empty frames into the SAE sample.

The model has two independent outputs, ``p_black`` and ``p_tunnel``.  A
panorama is rejected when any of its four views is black, or when at least two
views have a high tunnel probability.  The latter avoids dropping a valid
night street because of one dark heading.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

RES = 96
FEATURES = (
    "mean", "std", "dark_frac", "near_black_frac", "lap_log", "edge_frac",
    "top_mean", "bottom_mean", "warm", "glow", "bright_frac", "entropy",
)
DEFAULT_MODEL = Path(__file__).with_name("qc_model.joblib")


def _array(image) -> np.ndarray:
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB").resize((RES, RES)), np.float32)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {arr.shape}")
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).convert("RGB").resize(
        (RES, RES)), np.float32)


def features(image) -> np.ndarray:
    """Return one fixed-size, inexpensive quality descriptor."""
    from scipy import ndimage

    rgb = _array(image)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = ndimage.laplace(lum)
    gx = np.diff(lum, axis=1)
    gy = np.diff(lum, axis=0)
    hist, _ = np.histogram(lum, bins=24, range=(0, 255), density=False)
    p = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    entropy = float(-(p * np.log(p + 1e-8)).sum())
    bright = float(lum.mean())
    high = lum[lum >= np.percentile(lum, 95)]
    return np.asarray((
        bright / 255.0,
        float(lum.std()) / 255.0,
        float((lum < 20).mean()),
        float((lum < 8).mean()),
        float(np.log1p(lap.var())),
        float((float((np.abs(gx) > 18).mean()) +
               float((np.abs(gy) > 18).mean())) * 0.5),
        float(lum[: RES // 4].mean()) / 255.0,
        float(lum[-RES // 4 :].mean()) / 255.0,
        float(rgb[..., 0].mean() - rgb[..., 2].mean()) / 255.0,
        float(high.mean() - bright) / 255.0,
        float((lum > 245).mean()),
        entropy / np.log(24.0),
    ), dtype=np.float32)


def rule_flags(x: np.ndarray) -> tuple[bool, bool]:
    """Conservative weak labels: ``(black, tunnel)``."""
    mean, std, dark, near_black, lap_log, edge, top, _bottom, warm, glow, _sat, _ent = x
    black = bool(mean * 255.0 < 14.0 or dark > 0.97 or std * 255.0 < 6.0)
    tunnel = bool(
        not black
        and mean * 255.0 < 100.0
        and top * 255.0 < 95.0
        and lap_log < np.log1p(280.0)
        and (warm > 4.0 / 255.0 or glow > 42.0 / 255.0)
    )
    return black, tunnel


class QualityGate:
    """Score four PIL views and return a panorama-level keep/drop decision."""

    def __init__(self, model_path: str | os.PathLike | None = None,
                 black_threshold: float = 0.65, tunnel_threshold: float = 0.72,
                 tunnel_views: int = 2):
        self.model_path = Path(model_path or os.environ.get("QC_MODEL", DEFAULT_MODEL))
        self.black_threshold = float(black_threshold)
        self.tunnel_threshold = float(tunnel_threshold)
        self.tunnel_views = int(tunnel_views)
        self.black = self.tunnel = None
        self.model_loaded = False
        if self.model_path.exists():
            try:
                import joblib
                bundle = joblib.load(self.model_path)
                if tuple(bundle.get("features", ())) != FEATURES:
                    raise ValueError("quality model feature schema mismatch")
                self.black = bundle.get("black")
                self.tunnel = bundle.get("tunnel")
                self.model_loaded = self.black is not None and self.tunnel is not None
            except Exception:
                self.black = self.tunnel = None

    def score_view(self, image) -> tuple[float, float, bool, bool]:
        x = features(image)
        hard_black, hard_tunnel = rule_flags(x)
        if self.model_loaded:
            X = x.reshape(1, -1)
            p_black = float(self.black.predict_proba(X)[0, 1])
            p_tunnel = float(self.tunnel.predict_proba(X)[0, 1])
        else:
            p_black = float(hard_black)
            p_tunnel = float(hard_tunnel)
        # The hard black decision is intentionally never softened by a model.
        return p_black, p_tunnel, hard_black, hard_tunnel

    def score_views(self, views: Sequence) -> dict:
        scores = [self.score_view(v) for v in views]
        p_black = np.asarray([x[0] for x in scores], dtype=np.float32)
        p_tunnel = np.asarray([x[1] for x in scores], dtype=np.float32)
        hard_black = np.asarray([x[2] for x in scores], dtype=bool)
        hard_tunnel = np.asarray([x[3] for x in scores], dtype=bool)
        black = bool(hard_black.any() or (p_black >= self.black_threshold).any())
        # Learned scores are useful for generalization, but a high-confidence
        # deterministic tunnel signature must remain effective even when the
        # weakly supervised checkpoint has few real tunnel labels.
        tunnel = bool(
            (p_tunnel >= self.tunnel_threshold).sum() >= self.tunnel_views
            or hard_tunnel.sum() >= self.tunnel_views
        )
        return {
            "p_black": p_black,
            "p_tunnel": p_tunnel,
            "black": black,
            "tunnel": tunnel,
            "is_bad": black or tunnel,
            "model": self.model_loaded,
        }

    def filter_loaded(self, loaded: Iterable[tuple[str, Sequence]]) -> tuple[list, dict]:
        kept, stats = [], {"candidates": 0, "black": 0, "tunnel": 0, "kept": 0}
        for pid, views in loaded:
            stats["candidates"] += 1
            result = self.score_views(views)
            if result["black"]:
                stats["black"] += 1
            if result["tunnel"]:
                stats["tunnel"] += 1
            if result["is_bad"]:
                continue
            kept.append((pid, views)); stats["kept"] += 1
        return kept, stats


def train(paths: Sequence[str | os.PathLike], output: str | os.PathLike = DEFAULT_MODEL,
          seed: int = 0) -> dict:
    """Train the tiny weakly supervised gate and save it.

    Labels are generated only for high-confidence black/tunnel and clean
    examples.  This makes the checkpoint reproducible without pretending that
    unverified CLIP scores are ground-truth tunnel annotations.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    rng = np.random.default_rng(seed)
    X = np.asarray([features(p) for p in paths], dtype=np.float32)
    black = np.asarray([rule_flags(x)[0] for x in X], dtype=np.int64)
    tunnel = np.asarray([rule_flags(x)[1] for x in X], dtype=np.int64)

    # Most downloaded city samples contain no obvious defects.  Add a small,
    # deterministic synthetic calibration set in that case, so the learned
    # checkpoint is still usable before a human tunnel-label set is available.
    # These examples are only calibration anchors; the hard rules remain the
    # final authority for near-black frames.
    from PIL import Image, ImageDraw
    synth_images, synth_black, synth_tunnel = [], [], []
    for _ in range(256):
        level = float(rng.uniform(0.0, 10.0))
        synth_images.append(Image.fromarray(np.full((RES, RES, 3), level, np.uint8)))
        synth_black.append(1); synth_tunnel.append(0)
    for _ in range(256):
        rgb = np.zeros((RES, RES, 3), np.uint8)
        rgb[...] = (55, 42, 32)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        # Smooth dark enclosure with a few warm overhead lights.
        dr.rectangle((RES // 3, RES // 2, RES - 8, RES - 8), fill=(65, 50, 38))
        for x in (18, 48, 78):
            dr.ellipse((x, 30, x + 8, 38), fill=(225, 150, 65))
        synth_images.append(im); synth_black.append(0); synth_tunnel.append(1)
    Xs = np.asarray([features(im) for im in synth_images], dtype=np.float32)
    X = np.concatenate([X, Xs], axis=0)
    black = np.concatenate([black, np.asarray(synth_black, dtype=np.int64)])
    tunnel = np.concatenate([tunnel, np.asarray(synth_tunnel, dtype=np.int64)])
    models = {}
    for name, y in (("black", black), ("tunnel", tunnel)):
        if np.unique(y).size < 2:
            raise RuntimeError(f"cannot train {name} classifier: only one weak-label class")
        clf = HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.08, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=seed,
        )
        clf.fit(X, y); models[name] = clf
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": FEATURES, **models}, out)
    return {"images": len(paths), "black_labels": int(black.sum()),
            "tunnel_labels": int(tunnel.sum()), "path": str(out)}


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Train the tiny weakly supervised QC gate")
    parser.add_argument("--glob", required=True, help="image glob, for example '/nas/.../*_0.jpg'")
    parser.add_argument("--output", default=str(DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=0)
    cli = parser.parse_args()
    paths = sorted(glob.glob(cli.glob, recursive=True))
    if cli.limit:
        paths = paths[:cli.limit]
    if not paths:
        parser.error("the glob matched no images")
    print(train(paths, cli.output))
