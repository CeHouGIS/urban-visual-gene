#!/usr/bin/env python3
"""Recompute panorama-level community PPMI with the unchanged Top-N rule."""
from __future__ import annotations
import scripts._env  # noqa: F401
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, l2_rows

def statistics(counts:np.ndarray,top_n:int,support:int):
    n,k=counts.shape; integer=np.asarray(counts,dtype=np.int32); keep=min(top_n,k); top=np.argpartition(-integer,keep-1,axis=1)[:,:keep]
    presence=np.zeros((n,k),dtype=np.uint8); presence[np.arange(n)[:,None],top]=integer[np.arange(n)[:,None],top]>0; marg=presence.sum(0).astype(np.int64); pair=presence.T.astype(np.int64)@presence.astype(np.int64)
    expected=marg[:,None]*marg[None,:]/max(n,1); raw=np.log((pair+.5)/(expected+.5)); ppmi=np.maximum(raw,0); ppmi[pair<support]=0; np.fill_diagonal(ppmi,0)
    return marg,pair,ppmi

def long_table(names,marg,pair,ppmi,n,support):
    rows=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            rows.append({'community_i':names[i],'community_j':names[j],'panorama_support':int(pair[i,j]),'marginal_i':int(marg[i]),'marginal_j':int(marg[j]),'ppmi':float(ppmi[i,j]),'passes_support':bool(pair[i,j]>=support)})
    return pd.DataFrame(rows)

def run(config:dict,force:bool=False)->dict:
    assert_safe_affinity(); ensure_layout(config)
    cache=config['paths']['cache_root']; data=config['paths']['paper_data_root']; counts_dir=cache/'downstream/panorama_community_counts'
    membership=pd.read_csv(data/'leiden/latent_to_fine.csv'); names=sorted(membership.fine_community.unique()); files=sorted(counts_dir.glob('*.u16.npy')); top_n=int(config['downstream']['occurrence_top_n'])
    arrays=[np.asarray(np.load(f,mmap_mode='r'),dtype=np.uint16) for f in files]; total=sum(len(x) for x in arrays); support=max(int(config['downstream']['minimum_pair_support_absolute']),math.ceil(float(config['downstream']['minimum_pair_support_fraction'])*total))
    # Only 296k x 39 uint16 (~23 MB), safe to combine here.
    global_counts=np.concatenate(arrays); marg,pair,ppmi=statistics(global_counts,top_n,support); table=long_table(names,marg,pair,ppmi,total,support)
    table.to_csv(data/'downstream/global_ppmi.csv',index=False); table[['community_i','community_j','panorama_support','marginal_i','marginal_j']].to_csv(data/'downstream/global_cooccurrence.csv',index=False)
    city_vectors=[]; cities=[]
    for index,(file,counts) in enumerate(zip(files,arrays),1):
        city=file.stem.split('__')[-1].replace('.u16',''); local_support=max(50,math.ceil(float(config['downstream']['minimum_pair_support_fraction'])*len(counts))); lm,lp,lppmi=statistics(counts,top_n,local_support)
        long_table(names,lm,lp,lppmi,len(counts),local_support).to_csv(data/'downstream/city_ppmi'/f'{file.stem.replace(".u16","")}.csv',index=False)
        city_vectors.append(lppmi[np.triu_indices(len(names),1)]); cities.append(city); print(f'PPMI [{index:02d}/30] {city}',flush=True)
    vectors=np.stack(city_vectors); similarity=l2_rows(vectors)@l2_rows(vectors).T; pd.DataFrame(similarity,index=cities,columns=cities).to_csv(data/'downstream/city_cooccurrence_similarity.csv')
    graph=nx.Graph(); graph.add_nodes_from(names); candidates=table.loc[table.ppmi.gt(0)].sort_values('ppmi',ascending=False); kept=set()
    for node in names:
        local=candidates.loc[candidates.community_i.eq(node)|candidates.community_j.eq(node)].head(10)
        for r in local.itertuples(index=False): kept.add(tuple(sorted((r.community_i,r.community_j))))
    lookup={(r.community_i,r.community_j):r for r in candidates.itertuples(index=False)}
    for a,b in sorted(kept):
        r=lookup.get((a,b)) or lookup.get((b,a)); graph.add_edge(a,b,weight=float(r.ppmi),support=int(r.panorama_support))
    nx.write_graphml(graph,data/'downstream/community_network.graphml')
    return {'panoramas':total,'communities':len(names),'occurrence_top_n':top_n,'minimum_global_support':support,'positive_supported_edges':int((table.ppmi>0).sum())}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--force',action='store_true'); a=p.parse_args(); print(run(load_config(a.config),a.force))
if __name__=='__main__': main()
