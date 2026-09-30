#!/usr/bin/env python3
"""Identify stable gamma plateaus and select emergent coarse/fine scales."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, save_json


def candidate_plateaus(frame: pd.DataFrame, adjacent: np.ndarray, threshold: float, min_points: int, max_k_range: int) -> list[dict]:
    # K=1 is a trivial deterministic partition, not an organizational scale.
    good=(frame.mean_seed_nmi.to_numpy()>=threshold) & (frame.n_communities.to_numpy()>1)
    intervals=[]
    for left in range(len(frame)-min_points+1):
        if not good[left]: continue
        right=left
        while right+1<len(frame) and good[right+1] and adjacent[right]>=threshold:
            right+=1
        best=None
        for end in range(left+min_points-1,right+1):
            local=frame.n_communities.iloc[left:end+1]
            if local.max()-local.min()<=max_k_range: best=end
            else: break
        if best is not None:
            local=frame.iloc[left:best+1]
            intervals.append({"start_index":left,"end_index":best,"gamma_min":float(local.gamma.iloc[0]),"gamma_max":float(local.gamma.iloc[-1]),"points":len(local),"k_min":int(local.n_communities.min()),"k_max":int(local.n_communities.max()),"mean_seed_nmi":float(local.mean_seed_nmi.mean()),"mean_adjacent_nmi":float(adjacent[left:best].mean())})
    return intervals


def select_scales(config: dict) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    cache=config["paths"]["cache_root"]; data=config["paths"]["paper_data_root"]; figures=config["paths"]["paper_figure_root"]
    frame=pd.read_csv(data/"leiden/resolution_sweep.csv"); arrays=np.load(cache/"leiden/partitions/all_seed_partitions.npz"); consensus=arrays["consensus"]
    adjacent=np.asarray([normalized_mutual_info_score(consensus[i],consensus[i+1]) for i in range(len(consensus)-1)])
    frame["nmi_to_next_gamma"]=np.r_[adjacent,np.nan]
    chosen_threshold=None; plateaus=[]
    settings=config["plateau"]
    for threshold in settings["thresholds"]:
        plateaus=candidate_plateaus(frame,adjacent,float(threshold),int(settings["min_points"]),int(settings["max_k_range"]))
        if plateaus: chosen_threshold=float(threshold); break
    # Remove contained duplicates, preferring wider intervals.
    unique=[]
    for item in sorted(plateaus,key=lambda x:(-x["points"],x["gamma_min"])):
        if not any(item["start_index"]>=x["start_index"] and item["end_index"]<=x["end_index"] for x in unique): unique.append(item)
    plateaus=sorted(unique,key=lambda x:x["gamma_min"])
    def representative(plateau):
        indices=np.arange(plateau["start_index"],plateau["end_index"]+1); target=np.median(frame.n_communities.iloc[indices]); score=frame.mean_seed_nmi.iloc[indices].to_numpy()-0.001*np.abs(frame.n_communities.iloc[indices].to_numpy()-target); idx=int(indices[np.argmax(score)])
        return {"gamma":float(frame.gamma.iloc[idx]),"n_communities":int(frame.n_communities.iloc[idx]),"index":idx,"plateau":plateau,"status":"stable"}
    selected={}
    if plateaus:
        selected["coarse"]=representative(plateaus[0])
        higher=[x for x in plateaus[1:] if x["k_min"]>plateaus[0]["k_max"]]
        if higher: selected["fine"]=representative(higher[-1])
    if "coarse" not in selected:
        idx=int(frame.mean_seed_nmi.idxmax()); selected["coarse"]={"gamma":float(frame.gamma.iloc[idx]),"n_communities":int(frame.n_communities.iloc[idx]),"index":idx,"plateau":None,"status":"exploratory_no_plateau"}
    if "fine" not in selected:
        candidates=frame.index[frame.n_communities>selected["coarse"]["n_communities"]].to_numpy(); idx=int(candidates[np.argmax(frame.mean_seed_nmi.iloc[candidates])]) if len(candidates) else selected["coarse"]["index"]
        selected["fine"]={"gamma":float(frame.gamma.iloc[idx]),"n_communities":int(frame.n_communities.iloc[idx]),"index":idx,"plateau":None,"status":"exploratory_secondary"}
    payload={"stability_threshold_used":chosen_threshold,"candidate_plateaus":plateaus,"coarse":selected["coarse"],"fine":selected["fine"],"community_numbers_prespecified":False}
    save_json(data/"leiden/selected_scales.json",payload); frame.to_csv(data/"leiden/resolution_sweep.csv",index=False)
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,constrained_layout=True)
    axes[0].plot(frame.gamma,frame.n_communities,marker="o",ms=3); axes[0].set_ylabel("Communities K")
    axes[1].plot(frame.gamma,frame.mean_seed_nmi,label="NMI"); axes[1].plot(frame.gamma,frame.mean_seed_ari,label="ARI",alpha=.75); axes[1].set_ylabel("Within-resolution stability"); axes[1].legend(frameon=False)
    axes[2].plot(frame.gamma.iloc[:-1],adjacent,marker="o",ms=3); axes[2].set(xlabel="Leiden resolution gamma",ylabel="Adjacent-resolution NMI")
    colors={"coarse":"#56B4E9","fine":"#E69F00"}
    for name,scale in selected.items():
        plateau=scale["plateau"]
        for ax in axes:
            if plateau: ax.axvspan(plateau["gamma_min"],plateau["gamma_max"],color=colors[name],alpha=.16,label=name)
            ax.axvline(scale["gamma"],color=colors[name],ls="--",lw=1)
    fig.suptitle("Leiden multiscale resolution landscape")
    fig.savefig(figures/"resolution_landscape.png",dpi=240); plt.close(fig)
    return payload


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); args=parser.parse_args(); print(select_scales(load_config(args.config)))


if __name__ == "__main__": main()
