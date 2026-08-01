from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DINOv3 feature extraction entrypoint.")
    parser.add_argument("--config", required=True, help="Experiment config path")
    args = parser.parse_args()
    raise SystemExit(
        f"{args.config}: dataset-specific image loading is not wired yet; "
        "use sae_experiments.backbones.DINOv3Backbone and feature_cache utilities directly."
    )


if __name__ == "__main__":
    main()

