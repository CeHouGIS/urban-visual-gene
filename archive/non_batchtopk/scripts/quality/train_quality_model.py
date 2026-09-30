"""Train a lightweight image-quality classifier on heuristic labels (no torch).

Weak supervision: heuristic_label() provides clear good/bad examples; a small
gradient-boosting model learns a smooth boundary, resolves the 'ambiguous' cases,
and outputs a calibrated bad-quality probability per image. Aggregates to a
per-pano quality flag for excluding bad panos before feature extraction.

  python -m scripts.quality.train_quality_model --city Vienna

Outputs:
  models/quality_model.joblib
  outputs/quality/<city>_image_quality.parquet   (pano_id, heading, bad_prob, ...)
  outputs/quality/<city>_pano_quality.parquet     (pano_id, bad_prob, n_bad, exclude)
  outputs/figures/quality_{importance,flagged}.png
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.quality.image_quality import FEAT_COLS, heuristic_label, _load

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
QDIR = Path("outputs/quality"); QDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["Vienna"])
    ap.add_argument("--bad-headings", type=int, default=1,
                    help="exclude a pano if >= this many headings are bad")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    import joblib

    parts = []
    for c in args.cities:
        d = pd.read_parquet(QDIR / f"{c}_features.parquet"); d["city"] = c
        parts.append(d)
    df = pd.concat(parts).reset_index(drop=True)
    lab = [heuristic_label(r)[0] for r in df[FEAT_COLS].to_dict("records")]
    df["heur"] = lab
    train = df[df["heur"].isin(["good", "bad"])].copy()
    y = (train["heur"] == "bad").astype(int).values
    X = train[FEAT_COLS].values
    print(f"labeled: good={int((y==0).sum())} bad={int((y==1).sum())} "
          f"ambiguous(dropped)={int((df['heur']=='ambiguous').sum())}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                         max_depth=4, random_state=42)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    print(classification_report(yte, (proba > 0.5).astype(int),
                                target_names=["good", "bad"], digits=3))
    print("val ROC-AUC:", round(roc_auc_score(yte, proba), 4))
    joblib.dump({"model": clf, "features": FEAT_COLS}, "models/quality_model.joblib")

    # score ALL images (incl. ambiguous)
    df["bad_prob"] = clf.predict_proba(df[FEAT_COLS].values)[:, 1]
    df["bad"] = (df["bad_prob"] > args.threshold).astype(int)

    for c in args.cities:
        dc = df[df["city"] == c]
        dc.to_parquet(QDIR / f"{c}_image_quality.parquet", index=False)
        pano = (dc.groupby("pano_id")
                  .agg(bad_prob_max=("bad_prob", "max"),
                       bad_prob_mean=("bad_prob", "mean"),
                       n_bad=("bad", "sum"), n_head=("bad", "size")).reset_index())
        pano["exclude"] = pano["n_bad"] >= args.bad_headings
        pano.to_parquet(QDIR / f"{c}_pano_quality.parquet", index=False)
        print(f"{c}: panos={len(pano)} excluded(>= {args.bad_headings} bad head)="
              f"{int(pano['exclude'].sum())} ({pano['exclude'].mean()*100:.1f}%)")

    # ── figures ──────────────────────────────────────────────────────────────
    imp = pd.Series(clf.feature_importances_ if hasattr(clf, "feature_importances_")
                    else np.zeros(len(FEAT_COLS)), index=FEAT_COLS)
    if imp.sum() == 0:  # HistGBC has no feature_importances_; use permutation
        from sklearn.inspection import permutation_importance
        imp = pd.Series(permutation_importance(clf, Xte, yte, n_repeats=5,
                        random_state=42).importances_mean, index=FEAT_COLS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.sort_values().plot.barh(ax=ax, color="#d62728")
    ax.set_title(f"{args.city} — quality-model feature importance")
    fig.tight_layout(); fig.savefig(FIG / "quality_importance.png", dpi=130); plt.close(fig)

    # montage of the most-confidently-bad images (visual sanity check)
    worst = df.sort_values("bad_prob", ascending=False).head(24)
    fig, axes = plt.subplots(4, 6, figsize=(13, 9)); axes = axes.ravel()
    for ax, (_, r) in zip(axes, worst.iterrows()):
        ax.axis("off")
        try:
            rgb = _load(r["city"], r["pano_id"], int(r["heading"]))
            ax.imshow(rgb)
            ax.set_title(f"{r['city'][:2]} p={r['bad_prob']:.2f}", fontsize=8)
        except Exception:
            pass
    fig.suptitle("images flagged worst quality (top 24)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "quality_flagged.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved models/quality_model.joblib + outputs/figures/quality_*.png")


if __name__ == "__main__":
    main()
