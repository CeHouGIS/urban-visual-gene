#!/usr/bin/env python3
"""Map all 14x56 panorama winners to fixed Leiden fine communities."""
from __future__ import annotations
import scripts._env  # noqa: F401
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, l2_rows

def run(config:dict,force:bool=False)->dict:
    assert_safe_affinity(); ensure_layout(config)
    cache=config['paths']['cache_root']; data=config['paths']['paper_data_root']; root=config['paths']['panorama_root']
    membership=pd.read_csv(data/'leiden/latent_to_fine.csv').sort_values('dimension_id')
    communities=sorted(membership.fine_community.unique()); ci={x:i for i,x in enumerate(communities)}
    mapping=np.asarray([ci[x] for x in membership.fine_community],dtype=np.uint8)
    counts_dir=cache/'downstream/panorama_community_counts'; counts_dir.mkdir(parents=True,exist_ok=True)
    totals=[]
    for index,pred in enumerate(sorted((root/'predictions').glob('*/full')),1):
        slug=pred.parent.name; winners=np.load(pred/'top1_dimensions.u16.npy',mmap_mode='r'); target=counts_dir/f'{slug}.u16.npy'
        if force or not target.exists():
            out=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint16,shape=(len(winners),len(communities)))
            for start in range(0,len(winners),512):
                block=mapping[np.asarray(winners[start:start+512],dtype=np.int64)].reshape(len(winners[start:start+512]),-1)
                for j,row in enumerate(block): out[start+j]=np.bincount(row,minlength=len(communities))
            out.flush()
        values=np.asarray(np.load(target,mmap_mode='r'),dtype=np.uint64)
        if not np.all(values.sum(axis=1)==784): raise ValueError(f'{slug}: panorama community counts do not sum to 784')
        totals.append((slug,values.sum(axis=0)))
        print(f'composition [{index:02d}/30] {slug}: {len(values):,}',flush=True)
    matrix=np.stack([x[1] for x in totals]).astype(float); matrix/=matrix.sum(axis=1,keepdims=True); cities=[x[0].split('__')[-1] for x in totals]
    wide=pd.DataFrame(matrix,columns=communities); wide.insert(0,'city',cities); wide.to_csv(data/'downstream/city_composition_wide.csv',index=False)
    wide.melt(id_vars='city',var_name='community_id',value_name='prevalence').to_csv(data/'downstream/city_prevalence.csv',index=False)
    cosine=l2_rows(matrix)@l2_rows(matrix).T; pd.DataFrame(cosine,index=cities,columns=cities).to_csv(data/'downstream/city_cosine_similarity.csv')
    js=np.zeros_like(cosine)
    for i in range(len(cities)):
        for j in range(i+1,len(cities)): js[i,j]=js[j,i]=jensenshannon(matrix[i],matrix[j],base=2)
    pd.DataFrame(js,index=cities,columns=cities).to_csv(data/'downstream/city_js_distance.csv')
    entropy=-(matrix*np.log2(np.maximum(matrix,1e-15))).sum(1); normalized=entropy/math.log2(len(communities))
    pd.DataFrame({'city':cities,'entropy_bits':entropy,'normalized_entropy':normalized}).to_csv(data/'downstream/city_entropy.csv',index=False)
    return {'cities':len(cities),'communities':len(communities),'panoramas':int(sum(len(np.load(counts_dir/f'{x[0]}.u16.npy',mmap_mode='r')) for x in totals)),'row_sum_qa':float(np.max(np.abs(matrix.sum(1)-1)))}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--force',action='store_true'); a=p.parse_args(); print(run(load_config(a.config),a.force))
if __name__=='__main__': main()
