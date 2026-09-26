#!/usr/bin/env python3
"""Build the compact JSON used by the 512D × Mapillary-65 web explorer."""
from __future__ import annotations

import scripts._env  # noqa: F401  (must precede numpy / pandas)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "paper/data/semantic_alignment/feature_mae_mapillary_alignment_n60000"
)
DEFAULT_OUTPUT = ROOT / "dashboard/semantic_alignment/data.json"
DIMENSIONS = 512
CLASSES = 65


def build_payload(input_dir: Path) -> dict:
    distribution = pd.read_csv(input_dir / "dimension_semantic_distribution_65.csv")
    metrics = pd.read_csv(input_dir / "dimension_semantic_profiles.csv").sort_values(
        "dimension_id"
    )
    dimension_order = pd.read_csv(input_dir / "heatmap_dimension_order.csv")
    semantic_order = pd.read_csv(input_dir / "heatmap_semantic_order.csv")
    summary = json.loads((input_dir / "alignment_summary.json").read_text())

    profile_table = distribution.pivot(
        index="dimension_id", columns="class_id", values="semantic_fraction"
    ).reindex(index=np.arange(DIMENSIONS), columns=np.arange(CLASSES))
    if profile_table.shape != (DIMENSIONS, CLASSES) or profile_table.isna().any().any():
        raise ValueError("semantic distribution must form a complete 512 x 65 matrix")
    if metrics.dimension_id.tolist() != list(range(DIMENSIONS)):
        raise ValueError("dimension metrics must contain D000-D511 exactly once")

    class_rows = (
        distribution.drop_duplicates("class_id")
        .sort_values("class_id")
        .set_index("class_id")
    )
    semantic_order = semantic_order.set_index("class_id")
    dimension_order = dimension_order.set_index("dimension_id")

    classes = []
    for class_id in range(CLASSES):
        row = class_rows.loc[class_id]
        order = semantic_order.loc[class_id]
        classes.append(
            {
                "id": class_id,
                "name": str(row.class_name),
                "global_fraction": round(float(row.global_semantic_fraction), 8),
                "response_rank": int(order.heatmap_column),
                "dominant_dimension_count": int(order.dominant_dimension_count),
                "mean_fraction": round(float(order.mean_semantic_fraction), 8),
            }
        )

    dimensions = []
    for row in metrics.itertuples(index=False):
        order = dimension_order.loc[row.dimension_id]
        dimensions.append(
            {
                "id": int(row.dimension_id),
                "code": str(row.dimension),
                "response_rank": int(order.heatmap_row),
                "dominant_class_id": int(order.dominant_class_id),
                "dominant_fraction": round(float(order.dominant_fraction), 8),
                "top": [
                    {
                        "id": int(row.top1_class_id),
                        "name": str(row.top1_class),
                        "fraction": round(float(row.top1_share), 8),
                    },
                    {
                        "id": int(row.top2_class_id),
                        "name": str(row.top2_class),
                        "fraction": round(float(row.top2_share), 8),
                    },
                    {
                        "id": int(row.top3_class_id),
                        "name": str(row.top3_class),
                        "fraction": round(float(row.top3_individual_share), 8),
                    },
                ],
                "specificity": round(float(row.semantic_specificity), 8),
                "js_divergence": round(float(row.js_divergence_vs_global), 8),
                "seam_lift": round(float(row.seam_lift), 8),
                "assessment": str(row.assessment),
                "selected_images": int(row.selected_image_count),
                "selected_cities": int(row.selected_city_count),
            }
        )

    return {
        "version": 1,
        "method": summary["method"],
        "summary": {
            "sampled_panoramas": int(summary["sampled_panoramas"]),
            "valid_aligned_panoramas": int(summary["valid_aligned_panoramas"]),
            "dimensions": DIMENSIONS,
            "semantic_classes": CLASSES,
            "top_images_per_dimension": int(summary["top_images_per_dimension"]),
            "top_patches_per_dimension": int(summary["top_patches_per_dimension"]),
            "assessment_counts": summary["assessment_counts"],
        },
        "classes": classes,
        "dimensions": dimensions,
        "profiles": np.round(profile_table.to_numpy(dtype=np.float64), 6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dimensions": len(payload["dimensions"]),
                "classes": len(payload["classes"]),
                "matrix": [len(payload["profiles"]), len(payload["profiles"][0])],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
