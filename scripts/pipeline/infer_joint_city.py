"""Infer one city's activations with the ALREADY-TRAINED joint SAE basis.

Used to (re)run a single city through the shared basis without retraining — e.g.
CapeTown, whose inference OOM'd on the small-RAM GPU node during the joint run.
Run on a high-RAM (bigmem) node: the 774k-node x 4096-d context is large.

  python -m scripts.pipeline.infer_joint_city --city CapeTown
"""
import scripts._env  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.core.road_basis_model import RoadBasisAutoEncoder
from scripts.core.io_utils import save_report
from scripts.pipeline.stage5_infer_road_basis_activation import infer_activation

CITY_OUT = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITY_OUT))
    ap.add_argument("--joint-dir", default="outputs/transfer/joint")
    ap.add_argument("--hidden", type=int, default=512)
    args = ap.parse_args()

    jd = Path(args.joint_dir)
    basis = np.load(jd / "road_landscape_basis.npy")          # (K, D)
    K, D = basis.shape
    model = RoadBasisAutoEncoder(D=D, K=K, hidden=args.hidden)
    model.load_state_dict(torch.load(jd / "road_basis_model.pt", map_location="cpu"))
    model.eval()
    print(f"[infer] {args.city}: loaded joint model K={K} D={D}", flush=True)

    out = Path(CITY_OUT[args.city])
    (out / "stage_reports").mkdir(parents=True, exist_ok=True)
    ctx = pd.read_parquet(out / "road_context_features.parquet")
    act, r5 = infer_activation(ctx, model)
    act.to_parquet(out / "road_basis_activation.parquet", index=False)
    np.save(out / "road_landscape_basis.npy", basis)          # shared joint basis
    save_report(out / "stage_reports/stage5_report.json", r5)
    print(f"[infer] {args.city}: wrote {len(act):,} nodes, "
          f"median_active={r5['median_active_basis']:.1f}, recon={r5['mean_recon_error']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
