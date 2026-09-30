"""Plot the K-sweep: reconstruction elbow + vocabulary utilization (H1).

Reads outputs/ksweep/*/k_eval.json. Output: outputs/figures/k_sweep.png
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)


def main():
    rows = [json.loads(p.read_text())
            for p in sorted(Path("outputs/ksweep").glob("*/k_eval.json"))]
    df = pd.DataFrame(rows).drop_duplicates("K").sort_values("K")
    df.to_csv("outputs/sweep/k_sweep.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # (a) reconstruction error vs K — the elbow
    ax[0].plot(df["K"], df["recon_error"], "o-", color="#d62728", lw=2, ms=8)
    for _, r in df.iterrows():
        ax[0].annotate(f"{r.recon_error:.3f}", (r.K, r.recon_error),
                       textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(df["K"])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].set_title("(a) reconstruction error vs K"); ax[0].set_xlabel("K (bases)")
    ax[0].set_ylabel("mean recon error (1 - cos)"); ax[0].grid(True, alpha=0.3)

    # (b) capacity utilization: effective vocab / dominant / dead vs K
    ax[1].plot(df["K"], df["K"], "k--", lw=1, label="K (capacity)")
    ax[1].plot(df["K"], df["dominant"], "o-", color="#1f77b4", lw=2, label="dominant")
    ax[1].plot(df["K"], df["eff_vocab_90"], "s-", color="#2ca02c", lw=2,
               label="effective vocab (90%)")
    ax[1].plot(df["K"], df["dead"], "^-", color="#999999", lw=2, label="dead")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(df["K"])
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].set_title("(b) vocabulary utilization vs K")
    ax[1].set_xlabel("K (bases)"); ax[1].set_ylabel("# bases")
    ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)

    fig.suptitle("K-sweep — how many visual bases? (Vienna, covered nodes)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG / "k_sweep.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
