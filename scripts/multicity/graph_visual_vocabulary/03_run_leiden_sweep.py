#!/usr/bin/env python3
"""Run the multiseed Leiden resolution sweep and choose consensus partitions."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import itertools
import json
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .utils import DEFAULT_CONFIG, assert_safe_affinity, canonical_labels, ensure_layout, gamma_values, load_config, should_skip


def run_sweep(config: dict, force: bool = False) -> pd.DataFrame:
    assert_safe_affinity(); ensure_layout(config)
    cache=config["paths"]["cache_root"]; data=config["paths"]["paper_data_root"]
    target=data/"leiden/resolution_sweep.csv"
    if should_skip([target,cache/"leiden/partitions/all_seed_partitions.npz"],force): return pd.read_csv(target)
    edges=pd.read_csv(data/"graph/visual_graph_edges.csv")
    graph=ig.Graph(n=512,edges=list(zip(edges.source,edges.target)),directed=False); graph.es["weight"]=edges.weight_fused.tolist()
    gammas=gamma_values(config); n_seeds=int(config["leiden"]["n_seeds"])
    all_partitions=np.empty((len(gammas),n_seeds,512),dtype=np.int16); consensus=np.empty((len(gammas),512),dtype=np.int16); rows=[]
    for gamma_index,gamma in enumerate(gammas):
        qualities=[]; modularities=[]
        for seed in range(n_seeds):
            partition=leidenalg.find_partition(graph,leidenalg.RBConfigurationVertexPartition,weights="weight",resolution_parameter=float(gamma),seed=seed,n_iterations=-1)
            labels=canonical_labels(np.asarray(partition.membership)); all_partitions[gamma_index,seed]=labels
            qualities.append(float(partition.quality())); modularities.append(float(graph.modularity(labels.tolist(),weights="weight")))
        pairs=list(itertools.combinations(range(n_seeds),2)); ari=np.asarray([adjusted_rand_score(all_partitions[gamma_index,a],all_partitions[gamma_index,b]) for a,b in pairs]); nmi=np.asarray([normalized_mutual_info_score(all_partitions[gamma_index,a],all_partitions[gamma_index,b]) for a,b in pairs])
        mean_to_others=np.zeros(n_seeds)
        for seed in range(n_seeds): mean_to_others[seed]=np.mean([normalized_mutual_info_score(all_partitions[gamma_index,seed],all_partitions[gamma_index,other]) for other in range(n_seeds) if other!=seed])
        chosen=int(np.argmax(mean_to_others)); consensus[gamma_index]=all_partitions[gamma_index,chosen]
        sizes=np.bincount(consensus[gamma_index]); p=sizes/sizes.sum()
        rows.append({"gamma":gamma,"n_communities":len(sizes),"community_sizes":json.dumps(sizes.tolist()),"representative_seed":chosen,"representative_mean_nmi":mean_to_others[chosen],"quality":qualities[chosen],"modularity":modularities[chosen],"normalized_size_entropy":float(-(p*np.log(p)).sum()/np.log(len(p))) if len(p)>1 else 0.0,"mean_seed_ari":ari.mean(),"std_seed_ari":ari.std(),"mean_seed_nmi":nmi.mean(),"std_seed_nmi":nmi.std()})
        print(f"Leiden gamma={gamma:.2f}: K={len(sizes)}, seed NMI={nmi.mean():.4f}",flush=True)
    np.savez_compressed(cache/"leiden/partitions/all_seed_partitions.npz",gammas=gammas,partitions=all_partitions,consensus=consensus)
    frame=pd.DataFrame(rows); frame.to_csv(target,index=False); return frame


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--force",action="store_true"); args=parser.parse_args(); print(run_sweep(load_config(args.config),args.force).tail())


if __name__ == "__main__": main()
