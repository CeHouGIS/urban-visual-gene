#!/usr/bin/env python3
"""Export selected memberships, cross-scale associations and graph diagnostics."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .utils import DEFAULT_CONFIG, assert_safe_affinity, canonical_labels, ensure_layout, load_config, save_json


def reorder_by_size(labels: np.ndarray) -> np.ndarray:
    counts=pd.Series(labels).value_counts(); order=counts.index.tolist(); mapping={old:new for new,old in enumerate(order)}; return np.asarray([mapping[x] for x in labels],dtype=np.int16)


def metrics_for(labels: np.ndarray, graph: sparse.csr_matrix, prefix: str) -> pd.DataFrame:
    rows=[]; total_degree=np.asarray(graph.sum(axis=1)).ravel()
    for community in range(labels.max()+1):
        members=np.flatnonzero(labels==community); sub=graph[np.ix_(members,members)]; internal=float(sub.sum()/2); volume=float(total_degree[members].sum()); external=volume-2*internal; possible=len(members)*(len(members)-1)/2
        internal_strength=np.asarray(sub.sum(axis=1)).ravel(); medoid=int(members[np.argmax(internal_strength)])
        rows.append({"community_id":f"{prefix}{community+1:03d}","community_index":community,"community_size":len(members),"medoid_dimension":f"D{medoid:03d}","medoid_dimension_id":medoid,"internal_weight":internal,"mean_internal_edge_weight":float(sub.data.mean()) if sub.nnz else 0.0,"mean_external_weight_per_node":external/max(len(members),1),"internal_external_weight_ratio":internal/max(external,1e-12),"conductance":external/max(min(volume,2*float(graph.sum()/2)-volume),1e-12),"weighted_density":internal/max(possible,1)})
    return pd.DataFrame(rows)


def export(config: dict) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    cache=config["paths"]["cache_root"]; data=config["paths"]["paper_data_root"]
    selected=json.load(open(data/"leiden/selected_scales.json")); arrays=np.load(cache/"leiden/partitions/all_seed_partitions.npz"); consensus=arrays["consensus"]
    coarse=reorder_by_size(consensus[int(selected["coarse"]["index"])]); fine=reorder_by_size(consensus[int(selected["fine"]["index"])]); graph=sparse.load_npz(cache/"graph/visual_graph_fused.npz")
    coarse_ids=np.asarray([f"C{x+1:03d}" for x in coarse]); fine_ids=np.asarray([f"F{x+1:03d}" for x in fine])
    pd.DataFrame({"dimension_id":range(512),"dimension":[f"D{x:03d}" for x in range(512)],"coarse_community":coarse_ids}).to_csv(data/"leiden/latent_to_coarse.csv",index=False)
    pd.DataFrame({"dimension_id":range(512),"dimension":[f"D{x:03d}" for x in range(512)],"fine_community":fine_ids}).to_csv(data/"leiden/latent_to_fine.csv",index=False)
    coarse_metrics=metrics_for(coarse,graph,"C"); fine_metrics=metrics_for(fine,graph,"F"); coarse_metrics.to_csv(data/"validation/coarse_community_graph_metrics.csv",index=False); fine_metrics.to_csv(data/"validation/community_graph_metrics.csv",index=False)
    associations=[]
    for f in range(fine.max()+1):
        fm=fine==f; candidates=[]
        for c in range(coarse.max()+1):
            cm=coarse==c; intersection=int((fm&cm).sum()); union=int((fm|cm).sum()); candidates.append((intersection/max(union,1),intersection/max(fm.sum(),1),c,intersection))
        jaccard,purity,c,intersection=max(candidates)
        associations.append({"fine_community":f"F{f+1:03d}","best_coarse_id":f"C{c+1:03d}","jaccard_overlap":jaccard,"purity":purity,"intersection_dimensions":intersection,"fine_size":int(fm.sum())})
    pd.DataFrame(associations).to_csv(data/"leiden/fine_to_coarse_association.csv",index=False)
    payload={"coarse":{"gamma":selected["coarse"]["gamma"],"n_communities":int(coarse.max()+1)},"fine":{"gamma":selected["fine"]["gamma"],"n_communities":int(fine.max()+1)},"strictly_nested":bool(all(x["purity"]==1 for x in associations))}
    save_json(data/"leiden/final_partition_summary.json",payload); return payload


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); args=parser.parse_args(); print(export(load_config(args.config)))


if __name__ == "__main__": main()
