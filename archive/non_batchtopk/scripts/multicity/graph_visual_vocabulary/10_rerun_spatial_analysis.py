#!/usr/bin/env python3
"""Recompute grid homogeneity, local continuity and Moran's I."""
from __future__ import annotations
import scripts._env  # noqa: F401
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, l2_rows

EARTH_RADIUS=6378137.0
def web_mercator(lon,lat):
    x=EARTH_RADIUS*np.deg2rad(lon); y=EARTH_RADIUS*np.log(np.tan(np.pi/4+np.deg2rad(np.clip(lat,-85.0511,85.0511))/2)); return x,y

def grid_values(meta,counts,size,minimum):
    x,y=web_mercator(meta.lon.to_numpy(),meta.lat.to_numpy()); gx=np.floor(x/size).astype(np.int64); gy=np.floor(y/size).astype(np.int64)
    key=pd.MultiIndex.from_arrays([gx,gy]); codes,unique=pd.factorize(key,sort=True); ng=len(unique); summed=np.zeros((ng,counts.shape[1]),dtype=np.float64); np.add.at(summed,codes,counts); n=np.bincount(codes); keep=n>=minimum
    coords=np.asarray([[u[0],u[1]] for u in unique],dtype=np.int64)[keep]; values=summed[keep]; values/=np.maximum(values.sum(1,keepdims=True),1); return coords,values,n[keep]

def adjacency(coords):
    lookup={tuple(x):i for i,x in enumerate(coords)}; edges=[]
    for i,(x,y) in enumerate(coords):
        for target in ((x+1,y),(x,y+1)):
            if target in lookup: edges.append((i,lookup[target]))
    return np.asarray(edges,dtype=np.int64).reshape(-1,2)

def metrics(coords,values):
    n=len(values); normalized=l2_rows(values); homo=float((np.square(normalized.sum(0)).sum()-n)/(n*(n-1))) if n>1 else np.nan; edges=adjacency(coords); continuity=float(np.sum(normalized[edges[:,0]]*normalized[edges[:,1]],axis=1).mean()) if len(edges) else np.nan
    return homo,continuity,edges

def moran_rows(city,names,values,edges,permutations,rng):
    if not len(edges): return []
    n=len(values); rows=[]
    for j,name in enumerate(names):
        z=values[:,j]-values[:,j].mean(); denominator=float(z@z)
        if denominator<=1e-20: observed=np.nan; p=np.nan
        else:
            observed=float(n*np.sum(z[edges[:,0]]*z[edges[:,1]])/(len(edges)*denominator)); null=[]
            for _ in range(permutations):
                q=rng.permutation(z); null.append(float(n*np.sum(q[edges[:,0]]*q[edges[:,1]])/(len(edges)*(q@q))))
            null=np.asarray(null); p=float((1+(np.abs(null)>=abs(observed)).sum())/(permutations+1))
        rows.append({'city':city,'community_id':name,'morans_i':observed,'permutation_p_value':p,'n_grids':n,'n_adjacent_grid_pairs':len(edges)})
    return rows

def run(config:dict,force:bool=False)->dict:
    assert_safe_affinity(); ensure_layout(config)
    cache=config['paths']['cache_root']; data=config['paths']['paper_data_root']; root=config['paths']['panorama_root']; out=data/'downstream/spatial_metrics'; out.mkdir(parents=True,exist_ok=True)
    names=sorted(pd.read_csv(data/'leiden/latent_to_fine.csv').fine_community.unique()); sizes=config['downstream']['grid_sizes_m']; minimums=config['downstream']['minimum_panoramas']; primary=int(config['downstream']['primary_grid_size_m']); primary_min=int(config['downstream']['primary_minimum_panoramas']); rng=np.random.default_rng(int(config['random_seed']))
    summary=[]; sensitivity=[]; morans=[]; grid_rows=[]
    for index,file in enumerate(sorted((cache/'downstream/panorama_community_counts').glob('*.u16.npy')),1):
        slug=file.stem.replace('.u16',''); city=slug.split('__')[-1]; counts=np.asarray(np.load(file,mmap_mode='r'),dtype=np.float64); manifest=pd.read_parquet(root/'manifests'/f'{slug}.parquet').sort_values(['pano_index','direction_index']).drop_duplicates('pano_index').sort_values('pano_index')
        if len(manifest)!=len(counts): raise ValueError(f'{slug}: manifest/count mismatch')
        for size in sizes:
            for minimum in minimums:
                coords,values,nobs=grid_values(manifest,counts,int(size),int(minimum)); homo,continuity,edges=metrics(coords,values); sensitivity.append({'city':city,'grid_size_m':size,'minimum_panoramas':minimum,'n_valid_grids':len(values),'n_adjacent_grid_pairs':len(edges),'visual_homogeneity':homo,'local_visual_continuity':continuity})
                if int(size)==primary and int(minimum)==primary_min:
                    summary.append({'city':city,'n_valid_grids':len(values),'n_adjacent_grid_pairs':len(edges),'visual_homogeneity':homo,'visual_heterogeneity':1-homo,'local_visual_continuity':continuity})
                    morans.extend(moran_rows(city,names,values,edges,int(config['downstream']['moran_permutations']),rng))
                    for r,(coord,value,nx) in enumerate(zip(coords,values,nobs)):
                        for c,name in enumerate(names): grid_rows.append({'city':city,'grid_x':coord[0],'grid_y':coord[1],'panoramas':int(nx),'community_id':name,'prevalence':float(value[c])})
        print(f'spatial [{index:02d}/30] {city}',flush=True)
    pd.DataFrame(summary).to_csv(out/'city_visual_structure_summary.csv',index=False); pd.DataFrame(sensitivity).to_csv(out/'spatial_resolution_sensitivity.csv',index=False); pd.DataFrame(morans).to_csv(out/'community_morans_i.csv',index=False); pd.DataFrame(grid_rows).to_parquet(out/'grid_visual_composition.parquet',index=False,compression='zstd')
    return {'cities':len(summary),'grid_rows':len(grid_rows),'moran_tests':len(morans),'primary_grid_size_m':primary,'primary_minimum_panoramas':primary_min}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--force',action='store_true'); a=p.parse_args(); print(run(load_config(a.config),a.force))
if __name__=='__main__': main()
