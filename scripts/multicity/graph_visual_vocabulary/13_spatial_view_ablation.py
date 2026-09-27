#!/usr/bin/env python3
"""Run and visualize the E+D ablation against the primary E+D+P graph."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import itertools
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config


def consensus_sweep(graph: ig.Graph, gammas: np.ndarray, n_seeds: int):
    consensus=np.empty((len(gammas),512),dtype=np.int16); rows=[]
    for gi,gamma in enumerate(gammas):
        partitions=[]
        for seed in range(n_seeds):
            result=leidenalg.find_partition(graph,leidenalg.RBConfigurationVertexPartition,weights="weight",resolution_parameter=float(gamma),seed=seed,n_iterations=-1)
            partitions.append(np.asarray(result.membership,dtype=np.int16))
        pairs=list(itertools.combinations(range(n_seeds),2))
        nmi=np.asarray([normalized_mutual_info_score(partitions[a],partitions[b]) for a,b in pairs])
        ari=np.asarray([adjusted_rand_score(partitions[a],partitions[b]) for a,b in pairs])
        mean_to_others=np.asarray([np.mean([normalized_mutual_info_score(partitions[s],partitions[t]) for t in range(n_seeds) if t!=s]) for s in range(n_seeds)])
        chosen=int(np.argmax(mean_to_others)); consensus[gi]=partitions[chosen]
        rows.append({"gamma":float(gamma),"n_communities":int(len(np.unique(consensus[gi]))),"representative_seed":chosen,"mean_seed_nmi":float(nmi.mean()),"std_seed_nmi":float(nmi.std()),"mean_seed_ari":float(ari.mean()),"std_seed_ari":float(ari.std())})
        if (gi+1)%10==0: print(f"E+D sweep {gi+1}/{len(gammas)} gamma={gamma:.2f} K={rows[-1]['n_communities']}",flush=True)
    adjacent=np.asarray([normalized_mutual_info_score(consensus[i],consensus[i+1]) for i in range(len(consensus)-1)])
    frame=pd.DataFrame(rows); frame["nmi_to_next_gamma"]=np.r_[adjacent,np.nan]
    return frame,consensus


def graph_stats(graph: sparse.csr_matrix) -> dict:
    degree=np.diff(graph.indptr)
    return {"edges":int(graph.nnz//2),"density":float((graph.nnz//2)/(512*511/2)),"components":int(connected_components(graph,directed=False)[0]),"degree_min":int(degree.min()),"degree_mean":float(degree.mean()),"degree_max":int(degree.max())}


def plot_landscape(ed:pd.DataFrame,edp:pd.DataFrame,stats:dict,figure:Path):
    colors={"E+D":"#D55E00","E+D+P":"#0072B2"}; fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    ax=axes[0,0]; names=["E","D","P","E+D","E+D+P"]; values=[stats[x]["edges"] for x in names]; bars=ax.bar(names,values,color=["#999999","#777777","#BBBBBB",colors["E+D"],colors["E+D+P"]]); ax.bar_label(bars,fmt="%d",padding=3,fontsize=9); ax.set(ylabel="Undirected edges",title="A. Spatial view adds complementary graph relations"); ax.spines[['top','right']].set_visible(False)
    ax=axes[0,1]; ax.plot(ed.gamma,ed.n_communities,color=colors["E+D"],lw=2,label="E+D"); ax.plot(edp.gamma,edp.n_communities,color=colors["E+D+P"],lw=2,label="E+D+P"); ax.axvspan(2.65,2.75,color=colors["E+D+P"],alpha=.12,label="E+D+P stable plateau"); ax.scatter([1.70],[36],s=55,color=colors["E+D"],zorder=4); ax.scatter([2.70],[35],s=55,color=colors["E+D+P"],zorder=4); ax.annotate("E+D: γ=1.70, K=36\n(not stable)",(1.70,36),xytext=(1.05,55),arrowprops={"arrowstyle":"->","color":colors["E+D"]},fontsize=9); ax.annotate("E+D+P: γ=2.70, K=35\n(stable)",(2.70,35),xytext=(2.12,16),arrowprops={"arrowstyle":"->","color":colors["E+D+P"]},fontsize=9); ax.set(xlabel="Leiden resolution γ",ylabel="Communities K",title="B. Removing P accelerates graph fragmentation"); ax.legend(frameon=False,fontsize=9); ax.spines[['top','right']].set_visible(False)
    ax=axes[1,0]; ax.plot(ed.gamma,ed.mean_seed_nmi,color=colors["E+D"],lw=2,label="E+D"); ax.plot(edp.gamma,edp.mean_seed_nmi,color=colors["E+D+P"],lw=2,label="E+D+P"); ax.axhline(.85,color="#555555",ls="--",lw=1,label="0.85 plateau threshold"); ax.axvspan(2.65,2.75,color=colors["E+D+P"],alpha=.12); ax.set(xlabel="Leiden resolution γ",ylabel="Mean pairwise NMI across seeds",ylim=(0,1.03),title="C. Within-resolution stability"); ax.legend(frameon=False,fontsize=9); ax.spines[['top','right']].set_visible(False)
    ax=axes[1,1]; ax.plot(ed.gamma.iloc[:-1],ed.nmi_to_next_gamma.iloc[:-1],color=colors["E+D"],lw=2,label="E+D"); ax.plot(edp.gamma.iloc[:-1],edp.nmi_to_next_gamma.iloc[:-1],color=colors["E+D+P"],lw=2,label="E+D+P"); ax.axhline(.85,color="#555555",ls="--",lw=1,label="0.85 plateau threshold"); ax.axvspan(2.65,2.75,color=colors["E+D+P"],alpha=.12); ax.text(.04,.08,"No non-trivial E+D plateau\nat thresholds 0.90, 0.85, or 0.80",transform=ax.transAxes,color=colors["E+D"],fontsize=10); ax.set(xlabel="Leiden resolution γ",ylabel="NMI to next γ",ylim=(0,1.03),title="D. Cross-resolution stability"); ax.legend(frameon=False,fontsize=9); ax.spines[['top','right']].set_visible(False)
    fig.suptitle("Spatial-view ablation of the multiview visual-relation graph",fontsize=16); fig.savefig(figure,dpi=240,facecolor="white"); plt.close(fig)


def plot_graphs(ed_graph:sparse.csr_matrix,ed_labels:np.ndarray,edp_graph:sparse.csr_matrix,edp_labels:np.ndarray,positions:np.ndarray,figure:Path):
    ed_codes,_=pd.factorize(ed_labels,sort=True); edp_codes,_=pd.factorize(edp_labels,sort=True)
    overlap=np.zeros((ed_codes.max()+1,edp_codes.max()+1),dtype=np.int32)
    np.add.at(overlap,(ed_codes,edp_codes),1); source,target=linear_sum_assignment(-overlap)
    color_map={int(a):int(b) for a,b in zip(source,target)}; next_color=edp_codes.max()+1
    for value in np.unique(ed_codes):
        if int(value) not in color_map: color_map[int(value)]=next_color; next_color+=1
    aligned_ed=np.asarray([color_map[int(x)] for x in ed_codes]); color_max=max(int(aligned_ed.max()),int(edp_codes.max()))
    fig,axes=plt.subplots(1,2,figsize=(15,7.5),constrained_layout=True)
    for ax,title,graph,colors in [(axes[0],"E+D · γ=1.70 · K=36 · exploratory",ed_graph,aligned_ed),(axes[1],"E+D+P · γ=2.70 · K=35 · stable",edp_graph,edp_codes)]:
        tri=sparse.triu(graph,1).tocoo(); order=np.argsort(-tri.data)[:5000]; g=nx.Graph(); g.add_nodes_from(range(512)); g.add_weighted_edges_from((int(tri.row[i]),int(tri.col[i]),float(tri.data[i])) for i in order); pos={i:positions[i] for i in range(512)}; nx.draw_networkx_edges(g,pos,ax=ax,edge_color="#777777",alpha=.10,width=.25); nx.draw_networkx_nodes(g,pos,ax=ax,node_color=colors,cmap="turbo",vmin=0,vmax=color_max,node_size=18,linewidths=0); ax.set_title(title,fontsize=12); ax.axis("off")
    fig.suptitle("Same nodes and layout · E+D colors matched to E+D+P by maximum overlap",fontsize=15); fig.savefig(figure,dpi=240,facecolor="white"); plt.close(fig)


def run(config:dict,force:bool=False):
    assert_safe_affinity(); ensure_layout(config); cache=config["paths"]["cache_root"]; data=config["paths"]["paper_data_root"]; figures=config["paths"]["paper_figure_root"]
    E=sparse.load_npz(cache/"graph/graph_E.npz").tocsr(); D=sparse.load_npz(cache/"graph/graph_D.npz").tocsr(); P=sparse.load_npz(cache/"graph/graph_P.npz").tocsr(); ED=(E+D)/2; EDP=(E+D+P)/3
    csv_path=data/"validation/spatial_view_ablation_resolution_sweep.csv"; partition_path=cache/"validation/spatial_view_ablation_partitions.npz"
    gammas=np.round(np.arange(.05,3.001,.05),2)
    if force or not (csv_path.exists() and partition_path.exists()):
        tri=sparse.triu(ED,1).tocoo(); graph=ig.Graph(n=512,edges=list(zip(tri.row.tolist(),tri.col.tolist())),directed=False); graph.es["weight"]=tri.data.tolist(); sweep,consensus=consensus_sweep(graph,gammas,20); sweep.to_csv(csv_path,index=False); np.savez_compressed(partition_path,gammas=gammas,consensus=consensus)
    else: sweep=pd.read_csv(csv_path); consensus=np.load(partition_path)["consensus"]
    edp_sweep=pd.read_csv(data/"leiden/resolution_sweep.csv"); closest=int(np.lexsort((-sweep.mean_seed_nmi.to_numpy(),np.abs(sweep.n_communities.to_numpy()-35)))[0]); ed_labels=consensus[closest]; edp_labels=pd.read_csv(data/"leiden/latent_to_coarse.csv").sort_values("dimension_id").coarse_community.to_numpy()
    stats={"E":graph_stats(E),"D":graph_stats(D),"P":graph_stats(P),"E+D":graph_stats(ED),"E+D+P":graph_stats(EDP)}; pd.DataFrame(stats).T.rename_axis("graph").reset_index().to_csv(data/"validation/spatial_view_ablation_graph_stats.csv",index=False)
    plot_landscape(sweep,edp_sweep,stats,figures/"spatial_view_ablation_landscape.png"); positions=np.load(cache/"graph/visualization_positions.npy"); plot_graphs(ED,ed_labels,EDP,edp_labels,positions,figures/"spatial_view_ablation_graphs.png")
    return {"ed_closest_gamma":float(sweep.gamma.iloc[closest]),"ed_closest_k":int(sweep.n_communities.iloc[closest]),"landscape":str(figures/"spatial_view_ablation_landscape.png"),"graphs":str(figures/"spatial_view_ablation_graphs.png")}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--force",action="store_true"); args=parser.parse_args(); print(run(load_config(args.config),args.force))
if __name__=="__main__": main()
