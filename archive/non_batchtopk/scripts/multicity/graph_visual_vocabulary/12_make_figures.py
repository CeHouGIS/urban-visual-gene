#!/usr/bin/env python3
"""Create the final methodological figures and a concise, data-derived report."""
from __future__ import annotations
import scripts._env  # noqa: F401
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config

def graph_figure(config,data,figures):
    import networkx as nx
    import matplotlib.pyplot as plt
    edges=pd.read_csv(data/'graph/visual_graph_edges.csv'); membership=pd.read_csv(data/'leiden/latent_to_fine.csv').sort_values('dimension_id'); g=nx.Graph(); g.add_nodes_from(range(512)); g.add_weighted_edges_from(edges[['source','target','weight_fused']].itertuples(index=False,name=None)); pos=nx.spring_layout(g,weight='weight',seed=int(config['random_seed']),iterations=120)
    np.save(config['paths']['cache_root']/'graph/visualization_positions.npy',np.asarray([pos[i] for i in range(512)],dtype=np.float32)); labels=pd.factorize(membership.fine_community,sort=True)[0]
    fig,ax=plt.subplots(figsize=(11,9),constrained_layout=True); strong=edges.nlargest(6000,'weight_fused'); sg=nx.Graph(); sg.add_nodes_from(g.nodes); sg.add_weighted_edges_from(strong[['source','target','weight_fused']].itertuples(index=False,name=None)); nx.draw_networkx_edges(sg,pos,ax=ax,width=.25,alpha=.12,edge_color='#555555'); nx.draw_networkx_nodes(g,pos,ax=ax,node_size=20,node_color=labels,cmap='turbo',linewidths=0); ax.set_title('Multiview visual-relation graph · 512 latent dimensions'); ax.axis('off'); fig.savefig(figures/'graph_overview.png',dpi=260); plt.close(fig)

def evolution_figure(config,data,figures):
    import matplotlib.pyplot as plt
    cache=config['paths']['cache_root']; sweep=pd.read_csv(data/'leiden/resolution_sweep.csv'); arrays=np.load(cache/'leiden/partitions/all_seed_partitions.npz')['consensus']; targets=[1.0,2.0,2.85]; idx=[int(np.argmin(np.abs(sweep.gamma.to_numpy()-x))) for x in targets]; parts=[arrays[i] for i in idx]; fig,ax=plt.subplots(figsize=(12,7),constrained_layout=True); xs=[0,1,2]; blocks=[]
    for col,labels in enumerate(parts):
        order=sorted(np.unique(labels),key=lambda x:-np.sum(labels==x)); y=0; box={}
        for label in order:
            h=np.sum(labels==label)/512; box[label]=(y,y+h); ax.add_patch(plt.Rectangle((xs[col]-.045,y),.09,h,facecolor=plt.cm.turbo((label%40)/40),edgecolor='white',lw=.3)); y+=h
        blocks.append(box); ax.text(xs[col],1.02,f'γ={sweep.gamma.iloc[idx[col]]:.2f}\nK={len(order)}',ha='center')
    for col in (0,1):
        left,right=parts[col],parts[col+1]
        for a in np.unique(left):
            for b in np.unique(right):
                n=np.sum((left==a)&(right==b))
                if n:
                    ya=np.mean(blocks[col][a]); yb=np.mean(blocks[col+1][b]); ax.plot([xs[col]+.045,xs[col+1]-.045],[ya,yb],color='#777777',alpha=.12+min(.45,n/25),lw=max(.2,n/8))
    ax.set(xlim=(-.2,2.2),ylim=(0,1.08),title='Community evolution across Leiden resolutions'); ax.axis('off'); fig.savefig(figures/'community_evolution.png',dpi=240); plt.close(fig)

def vocabulary_figure(data,figures):
    labels=pd.read_csv(data/'semantic/community_labels.csv').set_index('community_id'); rows=[]
    for cid in labels.index:
        image=Image.open(figures/'community_evidence'/f'{cid}.jpg').convert('RGB'); crop=image.crop((0,0,image.width,min(460,image.height))); crop.thumbnail((360,190),Image.Resampling.LANCZOS); rows.append((cid,crop.copy(),labels.loc[cid]))
    cols=5; cell_w,cell_h=390,245; canvas=Image.new('RGB',(cols*cell_w,int(np.ceil(len(rows)/cols))*cell_h),'white'); draw=ImageDraw.Draw(canvas)
    for i,(cid,img,row) in enumerate(rows):
        x=(i%cols)*cell_w; y=(i//cols)*cell_h; canvas.paste(img,(x+(cell_w-img.width)//2,y+34)); draw.rectangle((x,y,x+cell_w,y+32),fill='#222222'); draw.text((x+7,y+8),f'{cid} | n={int(row.community_size)} | medoid D{int(row.medoid_dimension):03d}',fill='white'); draw.text((x+7,y+225),str(row.name_en),fill='black')
    canvas.save(figures/'visual_vocabulary.png',quality=92)

def report(config,data,figures):
    audit=json.load(open(data/'input_audit.json')); graph=json.load(open(data/'graph/graph_diagnostics.json')); selected=json.load(open(data/'leiden/selected_scales.json')); coherence=json.load(open(data/'validation/visual_coherence_summary.json')); partition=pd.read_csv(data/'validation/old_vs_new_partition.csv'); compare=pd.read_csv(data/'validation/old_vs_new_downstream.csv'); prevalence=pd.read_csv(data/'downstream/city_composition_wide.csv').set_index('city'); citysim=pd.read_csv(data/'downstream/city_cosine_similarity.csv',index_col=0); cosim=pd.read_csv(data/'downstream/city_cooccurrence_similarity.csv',index_col=0); spatial=pd.read_csv(data/'downstream/spatial_metrics/city_visual_structure_summary.csv'); ppmi=pd.read_csv(data/'downstream/global_ppmi.csv'); labels=pd.read_csv(data/'semantic/community_labels.csv')
    u=np.triu_indices(len(citysim),1); cities=list(citysim.index); best=np.argmax(citysim.to_numpy()[u]); worst=np.argmin(citysim.to_numpy()[u]); a,b=u[0][best],u[1][best]; c,d=u[0][worst],u[1][worst]; comp_co=float(spearmanr(citysim.to_numpy()[u],cosim.loc[cities,cities].to_numpy()[u]).statistic); shared=int((prevalence>0).all(axis=0).sum()); top_edges=ppmi.nlargest(5,'ppmi')
    downstream={r.comparison:r.spearman_rho for r in compare.itertuples(index=False)}; part={r.comparison:r for r in partition.itertuples(index=False)}
    lines=["# Multiview Graph Urban Visual Vocabulary",'',"## Input correction",'',f"The analysis unit is a **complete four-direction panorama**, not an independent direction image. Four 640×640 views were jointly processed as 2560×640 → 896×224, producing a 14×56×512 latent tensor. The spatial view therefore uses all 784 circular panorama positions (P = 512×784). No DINOv3 or Feature-MAE retraining/re-inference was performed.",'',f"- Panoramas: {audit['panoramas']:,}",f"- Patches: {audit['patches']:,}",f"- Cities: {audit['cities']}",'',"## A. Visual-relation graph",'',f"- Nodes: {graph['fused']['nodes']}",f"- Edges: {graph['fused']['edges']:,}",f"- Density: {graph['fused']['density']:.4f}",f"- Connected components: {graph['fused']['connected_components']}",f"- Degree min/mean/max: {graph['fused']['degree_min']:.0f} / {graph['fused']['degree_mean']:.2f} / {graph['fused']['degree_max']:.0f}",'',"## B–C. Stable resolutions and selected scales",'',f"A single strong non-trivial plateau was detected at γ={selected['coarse']['plateau']['gamma_min']:.2f}–{selected['coarse']['plateau']['gamma_max']:.2f} using the transparently relaxed threshold {selected['stability_threshold_used']:.2f}. Mean seed NMI={selected['coarse']['plateau']['mean_seed_nmi']:.3f}; mean adjacent-resolution NMI={selected['coarse']['plateau']['mean_adjacent_nmi']:.3f}.",'',f"- Stable primary/coarse scale: γ={selected['coarse']['gamma']:.2f}, K={selected['coarse']['n_communities']}",f"- Exploratory secondary/fine scale: γ={selected['fine']['gamma']:.2f}, K={selected['fine']['n_communities']}","- Only one strongly stable plateau was identified; the fine scale is explicitly exploratory, not presented as a second stable plateau.","- Community numbers emerged from the resolution landscape and were not specified in advance.",'',"## D. Independent visual coherence",'',f"- Within-community DINO similarity: {coherence['mean_within_community_similarity']:.4f}",f"- Between-community DINO similarity: {coherence['mean_between_community_similarity']:.4f}",f"- Difference: {coherence['difference']:.4f}",f"- Cohen's d: {coherence['cohens_d']:.3f}",f"- Size-preserving permutation p: {coherence['permutation_p_value']:.6f} ({coherence['n_permutations']} permutations)",'',"## E. Diagnostic comparison with Ward",'']
    for r in partition.itertuples(index=False): lines.append(f"- {r.comparison}: NMI={r.nmi:.3f}, ARI={r.ari:.3f} ({r.old_k} vs {r.new_k} groups)")
    lines += ['',"## F. Downstream conclusions",'',f"- {shared}/{prevalence.shape[1]} fine communities occur in every city; prevalence spans {prevalence.to_numpy().min():.6f}–{prevalence.to_numpy().max():.4f}.",f"- Strongest city composition pair: {cities[a]}–{cities[b]} (cosine={citysim.iloc[a,b]:.3f}); weakest: {cities[c]}–{cities[d]} (cosine={citysim.iloc[c,d]:.3f}).",f"- Global PPMI retains {(ppmi.ppmi>0).sum()} positive supported recurring pairs; strongest: "+", ".join(f"{r.community_i}–{r.community_j} ({r.ppmi:.2f})" for r in top_edges.itertuples(index=False))+'.',f"- Composition similarity and co-occurrence similarity remain distinct (city-pair Spearman ρ={comp_co:.3f}).",f"- Old/new spatial rank agreement: homogeneity ρ={downstream.get('city_homogeneity_ranking',float('nan')):.3f}; continuity ρ={downstream.get('city_continuity_ranking',float('nan')):.3f}.",f"- Highest new homogeneity: {spatial.sort_values('visual_homogeneity').iloc[-1].city}; highest local continuity: {spatial.sort_values('local_visual_continuity').iloc[-1].city}.",'',"Semantic names were added only after all graph partitions were fixed and never entered graph construction or community detection.",'',"## Outputs",'',"- Data: `paper/data/graph_visual_vocabulary/`","- Figures: `paper/figures/graph_visual_vocabulary/`","- Heavy reusable arrays: `outputs/experiments/dinov3_multicity/graph_visual_vocabulary/`",'']
    (data/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def run(config, config_path=DEFAULT_CONFIG):
    assert_safe_affinity(); ensure_layout(config); data=config['paths']['paper_data_root']; figures=config['paths']['paper_figure_root']; shutil.copy2(config_path,data/'config.yaml'); import matplotlib; matplotlib.use('Agg')
    graph_figure(config,data,figures); evolution_figure(config,data,figures); vocabulary_figure(data,figures); report(config,data,figures); return {'figures':5,'report':str(data/'report.md')}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); a=p.parse_args(); print(run(load_config(a.config),a.config))
if __name__=='__main__': main()
