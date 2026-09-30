#!/usr/bin/env python3
"""Re-infer Stage-5 activations using the JOINTLY-trained basis (both cities in
one coordinate system), so cross-city PCA/UMAP/colour are genuinely comparable.

Isolated torch process (import scripts._env first to lock thread pools).
Writes road_basis_activation_joint.parquet next to each city's per-city version.
"""
import scripts._env  # noqa: F401  (locks BLAS/OpenMP thread pools to 1)
from pathlib import Path
import torch
import pandas as pd
from scripts.core.road_basis_model import RoadBasisAutoEncoder
from scripts.pipeline.stage5_infer_road_basis_activation import infer_activation

JOINT_MODEL = "outputs/transfer/joint/road_basis_model.pt"
CITIES = ({"Vienna": "outputs/dedup_bld/Vienna", "HongKong": "outputs/dedup_bld/HongKong"}
          if __import__("os").environ.get("USE_DEDUP_BLD")
          else {"Vienna": "outputs/sweep/Vienna_N5000", "HongKong": "outputs/sweep/HongKong_N2000"})


def main():
    model = RoadBasisAutoEncoder(D=3072, K=32, hidden=512)
    model.load_state_dict(torch.load(JOINT_MODEL, map_location="cpu"))
    model.eval()
    print("[joint-s5] loaded joint basis model")

    for city, run in CITIES.items():
        d = Path(run)
        ctx = pd.read_parquet(d / "road_context_features.parquet")
        print(f"[joint-s5] {city}: {len(ctx):,} nodes — inferring activations under joint basis")
        act_df, rep = infer_activation(ctx, model)
        out = d / "road_basis_activation_joint.parquet"
        act_df.to_parquet(out, index=False)
        print(f"[joint-s5] {city}: wrote {out}  median_active={rep['median_active_basis']:.1f}/32 "
              f"mean_recon={rep.get('mean_reconstruction_error', float('nan')):.3f}")


if __name__ == "__main__":
    main()
