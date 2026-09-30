#!/usr/bin/env python3
"""Per-node dominant basis under the K=128 SAE, for the map's '主导基·128' mode.

Writes dashboard/data/<city>_dom128.bin (Uint8, argmax over the 128 activations,
same node order as <city>_pos.bin). Reads road_basis_activation_k128.parquet.
  OMP_NUM_THREADS=1 python -m scripts.dash.build_dom128
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
CITIES = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}
ACT = [f"a_{i:03d}" for i in range(128)]


def main():
    for c, run in CITIES.items():
        A = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_k128.parquet"),
                            columns=ACT).to_numpy(np.float32)
        dom = A.argmax(1).astype(np.uint8)          # 0..127
        dom.tofile(os.path.join(OUT, f"{c}_dom128.bin"))
        print(f"[dom128] {c}: {len(dom):,} nodes, distinct dominant bases used = "
              f"{len(np.unique(dom))}", flush=True)


if __name__ == "__main__":
    main()
