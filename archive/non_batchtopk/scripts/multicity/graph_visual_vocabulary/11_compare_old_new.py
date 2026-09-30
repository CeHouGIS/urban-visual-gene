#!/usr/bin/env python3
"""Diagnostic comparison with the retained Ward partitions and downstream results."""
from __future__ import annotations
import scripts._env  # noqa: F401
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, save_json

def matrix(path):
    x=pd.read_csv(path,index_col=0); return x
def upper_corr(a,b):
    common=sorted(set(a.index)&set(b.index)); av=a.loc[common,common].to_numpy(); bv=b.loc[common,common].to_numpy(); u=np.triu_indices(len(common),1); return float(spearmanr(av[u],bv[u]).statistic),len(common)

def run(config:dict)->dict:
    assert_safe_affinity(); ensure_layout(config); data=config['paths']['paper_data_root']; figures=config['paths']['paper_figure_root']
    old=pd.read_csv(config['paths']['old_membership']).sort_values('feature_id'); coarse=pd.read_csv(data/'leiden/latent_to_coarse.csv').sort_values('dimension_id'); fine=pd.read_csv(data/'leiden/latent_to_fine.csv').sort_values('dimension_id')
    partition=[{'comparison':'old_Ward_64_vs_new_fine','old_k':old.fine_cluster.nunique(),'new_k':fine.fine_community.nunique(),'nmi':normalized_mutual_info_score(old.fine_cluster,fine.fine_community),'ari':adjusted_rand_score(old.fine_cluster,fine.fine_community)}, {'comparison':'old_Ward_32_vs_new_coarse','old_k':old.coarse_cluster.nunique(),'new_k':coarse.coarse_community.nunique(),'nmi':normalized_mutual_info_score(old.coarse_cluster,coarse.coarse_community),'ari':adjusted_rand_score(old.coarse_cluster,coarse.coarse_community)}]
    pd.DataFrame(partition).to_csv(data/'validation/old_vs_new_partition.csv',index=False)
    old_comp=pd.read_csv(config['paths']['old_city_composition']).set_index('city'); new_comp=pd.read_csv(data/'downstream/city_composition_wide.csv').set_index('city'); old_cos=pd.DataFrame((old_comp.to_numpy()/np.linalg.norm(old_comp,axis=1,keepdims=True))@(old_comp.to_numpy()/np.linalg.norm(old_comp,axis=1,keepdims=True)).T,index=old_comp.index,columns=old_comp.index); new_cos=matrix(data/'downstream/city_cosine_similarity.csv'); comp_corr,ncities=upper_corr(old_cos,new_cos)
    old_co=matrix(config['paths']['old_city_cooccurrence_similarity']); new_co=matrix(data/'downstream/city_cooccurrence_similarity.csv'); co_corr,_=upper_corr(old_co,new_co)
    # Lift both differently-sized PPMI matrices back to the shared 512 dimensions.
    old_ppmi=matrix(Path(config['paths']['old_city_cooccurrence_similarity']).parent/'global_element_ppmi.csv'); new_long=pd.read_csv(data/'downstream/global_ppmi.csv'); new_names=sorted(fine.fine_community.unique()); new_ppmi=pd.DataFrame(0.,index=new_names,columns=new_names)
    for r in new_long.itertuples(index=False): new_ppmi.loc[r.community_i,r.community_j]=new_ppmi.loc[r.community_j,r.community_i]=r.ppmi
    old_lookup={x:i for i,x in enumerate(old_ppmi.index)}; new_lookup={x:i for i,x in enumerate(new_ppmi.index)}; old_label=old.fine_cluster.to_numpy(); new_label=fine.fine_community.to_numpy(); old_lift=old_ppmi.to_numpy()[[old_lookup[x] for x in old_label]][:,[old_lookup[x] for x in old_label]]; new_lift=new_ppmi.to_numpy()[[new_lookup[x] for x in new_label]][:,[new_lookup[x] for x in new_label]]; u=np.triu_indices(512,1); ppmi_corr=float(spearmanr(old_lift[u],new_lift[u]).statistic)
    old_spatial=pd.read_csv(config['paths']['old_spatial_summary']); new_spatial=pd.read_csv(data/'downstream/spatial_metrics/city_visual_structure_summary.csv'); joined=old_spatial.merge(new_spatial,on='city',suffixes=('_old','_new')); hom_corr=float(spearmanr(joined.visual_homogeneity_old,joined.visual_homogeneity_new).statistic); cont_corr=float(spearmanr(joined.local_visual_continuity_old,joined.local_visual_continuity_new).statistic)
    downstream=pd.DataFrame([{'comparison':'city_composition_cosine_matrices','spearman_rho':comp_corr,'n_units':ncities},{'comparison':'city_cooccurrence_similarity_matrices','spearman_rho':co_corr,'n_units':ncities},{'comparison':'dimension_lifted_global_PPMI','spearman_rho':ppmi_corr,'n_units':512},{'comparison':'city_homogeneity_ranking','spearman_rho':hom_corr,'n_units':len(joined)},{'comparison':'city_continuity_ranking','spearman_rho':cont_corr,'n_units':len(joined)}]); downstream.to_csv(data/'validation/old_vs_new_downstream.csv',index=False)
    summary={'partitions':partition,'downstream':downstream.to_dict('records')}; save_json(data/'validation/old_vs_new_summary.json',summary)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True); p=pd.DataFrame(partition); axes[0].bar(np.arange(2)-.18,p.nmi,.36,label='NMI'); axes[0].bar(np.arange(2)+.18,p.ari,.36,label='ARI'); axes[0].set_xticks(range(2),['Fine','Coarse']); axes[0].set_ylim(0,1); axes[0].set_title('Ward vs graph communities'); axes[0].legend(frameon=False); axes[1].barh(downstream.comparison.str.replace('_',' '),downstream.spearman_rho,color='#4C78A8'); axes[1].axvline(0,color='black',lw=.7); axes[1].set_xlim(-1,1); axes[1].set_title('Downstream Spearman agreement'); fig.savefig(figures/'old_vs_new.png',dpi=240); plt.close(fig)
    return summary

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); a=p.parse_args(); print(json.dumps(run(load_config(a.config)),indent=2))
if __name__=='__main__': main()
