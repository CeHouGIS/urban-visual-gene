"""Reproducible Chapter 5 revision from frozen scores and cached evidence.

No fitting of DINO or SAE. UMAP is a descriptive projection of observed cells.
Run --phase umap separately from --phase plots to overlap plotting with UMAP.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from .semantic_22_figures import CLASSES, LABELS, STAGE
from .semantic_city_figures import ROOT, name, UNIT_NAMES, PARENTS, city_similarity_figure
from .followup import FOLLOWUP

OUT = ROOT/'experiments/vitl30_city_patterns/results/chapter5_revision'
ASSETS = Path('/root/code/HKUST-Thesis/img_gene/vitl30_w4096')
PATTERNS = FOLLOWUP/'stages/composition_patterns_screened'
OLD = ROOT/'experiments/vitl30_city_patterns/results/figures'
INTERNAL = ROOT/'experiments/vitl30_city_patterns/results/city_internal_compositions'
PALETTE = list(plt.get_cmap('tab20').colors)+['#545454','#b39a68']

def style():
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':12.5,
        'axes.labelsize':13,'axes.titlesize':13.5,'xtick.labelsize':12,
        'ytick.labelsize':12,'legend.fontsize':12,'pdf.fonttype':42,
        'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    OUT.mkdir(parents=True,exist_ok=True)

def save(fig,stem):
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    shutil.copy2(OUT/f'{stem}.pdf',ASSETS/f'{stem}.pdf')

def order(values):
    return leaves_list(linkage(values,method='average',metric='euclidean',optimal_ordering=True))

def city_labels(cities):
    return [name(c) for c in cities]

def umap_plot():
    import umap
    cells=pd.read_parquet(STAGE/'cell_class_shares.parquet')
    sample=[]
    for city,g in cells.groupby('city',sort=True):
        g=g.copy()
        g['selection_hash']=[hashlib.sha256(f'{city}/{x}/{y}'.encode()).hexdigest()
                             for x,y in zip(g.grid_x,g.grid_y)]
        sample.append(g.sort_values('selection_hash').head(1000))
    sample=pd.concat(sample,ignore_index=True)
    values=sample[list(CLASSES)].to_numpy()
    identity=hashlib.sha256(values.tobytes()+''.join(sample.selection_hash).encode()).hexdigest()
    cache=OUT/'umap_coordinates.npz'
    if cache.exists():
        data=np.load(cache)
        if str(data['identity'])!=identity: raise ValueError('UMAP sample changed')
        xy=data['xy']
    else:
        print(f'UMAP: {len(sample):,} cells in {sample.city.nunique()} cities',flush=True)
        xy=umap.UMAP(n_neighbors=30,min_dist=.1,metric='cosine',random_state=42,n_jobs=1).fit_transform(values)
        np.savez_compressed(cache,xy=xy,identity=identity)
    sample['umap_1']=xy[:,0];sample['umap_2']=xy[:,1]
    sample.to_parquet(OUT/'umap_sample.parquet',index=False)
    counts=sample.groupby('city').size().to_dict()
    (OUT/'umap_protocol.json').write_text(json.dumps({'cells':len(sample),'per_city_cap':1000,
        'counts':counts,'metric':'cosine','neighbors':30,'min_dist':.1,'seed':42,
        'selection':'sha256(city/grid_x/grid_y)','identity':identity},indent=2))
    fig=plt.figure(figsize=(10.5,6.7))
    ax=fig.add_axes([.07,.10,.61,.85])
    palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('tab20b').colors)[:10]
    # Interleave cities to avoid systematic overpainting of one city.
    draw=np.random.default_rng(42).permutation(len(sample))
    ids=pd.Categorical(sample.city,categories=sorted(counts)).codes
    ax.scatter(xy[draw,0],xy[draw,1],s=3.2,c=np.array(palette)[ids[draw]],alpha=.58,
               linewidths=0,rasterized=True)
    handles=[plt.Line2D([],[],marker='o',ls='',color=palette[i],markersize=5,
                       label=f'{name(city)} ({counts[city]:,})') for i,city in enumerate(sorted(counts))]
    fig.legend(handles=handles,loc='center right',bbox_to_anchor=(.998,.52),
               frameon=False,fontsize=12,labelspacing=.5)
    ax.set(xlabel='UMAP 1',ylabel='UMAP 2')
    save(fig,'Fig_ViTL30_Semantic_Cell_UMAP')
    print('UMAP complete',flush=True)

def city_plots(cells):
    means=cells.groupby('city')[list(CLASSES)].mean().sort_index()
    cities=means.index.tolist();core=means.to_numpy()
    diff=100*(core-(core.sum(axis=0)-core)/(len(cities)-1))
    z=linkage(diff,method='average',optimal_ordering=True)
    co=leaves_list(z)
    spatial_idx=CLASSES.index('spatial_encoding')
    ro=sorted((i for i in range(len(CLASSES)) if i != spatial_idx),
              key=lambda i:-core[:,i].mean())+[spatial_idx]
    fig=plt.figure(figsize=(11.2,8.7))
    ax=fig.add_axes([.265,.22,.665,.625])
    # Same physical unit in x and y gives square cells.
    im=ax.imshow(diff[np.ix_(co,ro)].T,aspect='equal',cmap='RdBu_r',
                 norm=TwoSlopeNorm(vcenter=0,vmin=-np.quantile(abs(diff),.98),vmax=np.quantile(abs(diff),.98)))
    ax.set_xticks(range(30),city_labels([cities[i] for i in co]),rotation=90)
    ax.set_yticks(range(22),[LABELS[CLASSES[i]] for i in ro])
    dx=fig.add_axes([.265,.86,.665,.09])
    dendrogram(z,ax=dx,no_labels=True,color_threshold=0,above_threshold_color='#53686a')
    dx.axis('off')
    cb=fig.add_axes([.95,.26,.016,.54]);fig.colorbar(im,cax=cb,label='Difference from other cities (percentage points)')
    pd.DataFrame({'city':[cities[i] for i in co]}).to_csv(OUT/'city_excess_order.csv',index=False)
    save(fig,'Fig_ViTL30_Semantic_City_Excess')

    fig,ax=plt.subplots(figsize=(10.5,6.5))
    bottom=np.zeros(30)
    for j in sorted(range(22),key=lambda j:-core[:,j].mean()):
        ax.bar(range(30),core[:,j]*100,bottom=bottom,color=PALETTE[j],width=.85,label=LABELS[CLASSES[j]])
        bottom+=core[:,j]*100
    ax.set_xticks(range(30),city_labels(cities),rotation=90)
    ax.set(ylabel='Share of positive SAE response (%)',ylim=(0,100),xlim=(-.7,29.7))
    fig.subplots_adjust(left=.075,right=.99,bottom=.23,top=.69)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,
               fontsize=12,columnspacing=1.,handlelength=1.3)
    save(fig,'Fig_ViTL30_Semantic_City_Composition')

    city_similarity_figure(cities, core, OUT)
    shutil.copy2(OUT/'Fig_ViTL30_Semantic_City_Similarity.pdf',
                 ASSETS/'Fig_ViTL30_Semantic_City_Similarity.pdf')

def pattern_plots():
    key=pd.read_csv(OLD/'selected_composition_key.csv')
    rates=pd.read_csv(PATTERNS/'city_distribution.csv')
    m=rates.pivot(index='city',columns='pattern_id',values='cell_weighted_rate')[key.pattern_id]*100
    rz=linkage(m.to_numpy(),method='average',optimal_ordering=True)
    cz=linkage(m.to_numpy().T,method='average',optimal_ordering=True)
    rr=leaves_list(rz);cc=leaves_list(cz)
    fig=plt.figure(figsize=(10.7,9.1))
    ax=fig.add_axes([.24,.28,.50,.59])
    im=ax.imshow(m.to_numpy()[np.ix_(rr,cc)],aspect='auto',cmap='YlGnBu',vmin=0)
    ax.set_yticks(range(30),city_labels(m.index[rr]))
    ax.set_xticks(range(12),key.display_id.iloc[cc],rotation=90)
    dx=fig.add_axes([.24,.89,.50,.08]);dendrogram(cz,ax=dx,no_labels=True,color_threshold=0,above_threshold_color='#53686a');dx.axis('off')
    dy=fig.add_axes([.01,.28,.06,.59]);dendrogram(rz,ax=dy,orientation='left',no_labels=True,color_threshold=0,above_threshold_color='#53686a');dy.invert_yaxis();dy.axis('off')
    cb=fig.add_axes([.77,.34,.02,.47]);fig.colorbar(im,cax=cb,label='Cell-balanced matched panoramas (%)')
    for i,row in key.iterrows():
        fig.text(.05+(i%2)*.48,.205-(i//2)*.031,
                 f'{row.display_id}  {row.english_units}',fontsize=12)
    save(fig,'Fig_CityPatterns_Current_Cities')
    (OUT/'pattern_cluster_order.json').write_text(json.dumps({'cities':m.index[rr].tolist(),
        'pattern_ids':key.pattern_id.iloc[cc].tolist(),'display_ids':key.display_id.iloc[cc].tolist(),
        'method':'average linkage, Euclidean distance on percentage rates'},indent=2))

    intervals=pd.read_csv(OLD/'selected_composition_city_intervals.csv')
    fig,axes=plt.subplots(1,6,figsize=(11.2,7.7),sharey=True)
    cities=sorted(m.index)
    for i,ax in enumerate(axes):
        ident=f'P{i+1:02d}';g=intervals[intervals.display_id==ident].set_index('city').loc[cities]
        ax.hlines(range(30),g.low*100,g.high*100,color='#82aaa5',lw=1.4)
        ax.scatter(g.rate*100,range(30),s=17,color='#287b77',zorder=3)
        ax.set_yticks(range(30),city_labels(cities));ax.set_title(ident,weight='bold')
        ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='x',labelsize=10)
    axes[0].invert_yaxis();fig.supxlabel('Cell-balanced matched panoramas (%)',fontsize=13)
    fig.subplots_adjust(left=.16,right=.99,bottom=.08,top=.94,wspace=.12)
    save(fig,'Fig_CityPatterns_Current_Intervals')

def study_map():
    import geopandas as gpd
    from pyproj import Transformer
    frame=pd.read_parquet(FOLLOWUP/'cells.parquet',columns=['city','grid_x','grid_y','utm_epsg'])
    points=[]
    for city,g in frame.groupby('city',sort=True):
        epsg=int(g.utm_epsg.iloc[0]);tr=Transformer.from_crs(epsg,4326,always_xy=True)
        lon,lat=tr.transform((g.grid_x.median()+.5)*1000,(g.grid_y.median()+.5)*1000)
        points.append((city,lon,lat))
    world=gpd.read_file('/usr/local/lib/python3.11/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp')
    fig=plt.figure(figsize=(10.7,7.1));ax=fig.add_axes([.01,.34,.98,.64])
    world.plot(ax=ax,color='#e5e8e6',edgecolor='white',linewidth=.35)
    ax.set(xlim=(-175,180),ylim=(-60,80));ax.axis('off')
    # Dense European labels receive a separate inset; every city name is on a map.
    europe=[p for p in points if -12<p[1]<30 and p[2]>35]
    ax.scatter([p[1] for p in points],[p[2] for p in points],s=19,c='#b64b3d',ec='white',lw=.4,zorder=5)
    offsets={'US/Chicago':(-21,13),'US/NewYork':(7,3),'US/Phoenix':(-35,-6),
        'Brazil/Brasília':(-20,-13),'Brazil/SaoPaulo':(-36,-17),'Brazil/RiodeJaneiro':(10,-5),
        'Korea/Seoul':(6,11),'China/Taipei':(13,1),'Bangladesh/Dhaka':(-37,11),
        'UAE/Dubai':(-30,5),'Singapore/Singapore':(-57,-8),'Indonesia/Yogyakarta':(-52,-24),
        'Australia/Perth':(-30,-13),'Australia/Melbourne':(-48,-24),'Australia/Sydney':(8,0),
        'SouthAfrica/Johannesburg':(13,0),'SouthAfrica/CapeTown':(-31,-17),
        'Nigeria/Lagos':(-27,-15),'Mexico/Monterrey':(-25,-17),
        'Argentina/BuenosAires':(-18,-20),'Russia/Moscow':(7,8)}
    for city,x,y in points:
        if (city,x,y) in europe:continue
        dx,dy=offsets.get(city,(5,5))
        ax.annotate(name(city),(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=13,
                    arrowprops={'arrowstyle':'-','color':'#727e7a','lw':.6},zorder=6)
    eu=fig.add_axes([.08,.015,.47,.34]);world.plot(ax=eu,color='#e5e8e6',edgecolor='white',linewidth=.5)
    eu.set(xlim=(-12,24),ylim=(36,62));eu.axis('off')
    eo={'Portugal/Lisbon':(-34,-4),'Spain/Barcelona':(12,-12),'France/Paris':(-42,-10),
        'Belgium/Brussels':(-70,3),'Belgium/Antwerp':(-65,17),'Germany/Hamburg':(12,0),
        'Austria/Vienna':(12,4),'Italy/Naples':(10,-10),'Sweden/Stockholm':(12,0)}
    for city,x,y in europe:
        eu.scatter([x],[y],s=20,c='#b64b3d',ec='white',lw=.4,zorder=5)
        eu.annotate(name(city),(x,y),xytext=eo.get(city,(4,4)),textcoords='offset points',fontsize=13,
                    arrowprops={'arrowstyle':'-','color':'#727e7a','lw':.6})
    fig.text(.61,.22,'30 study cities',fontsize=16,weight='bold',color='#314b4b')
    fig.text(.61,.14,'Complete four-direction panoramas\nCity positions shown for orientation',fontsize=12.5,linespacing=1.5)
    save(fig,'Fig_ViTL30_Study_Cities')

def workflow():
    fig,ax=plt.subplots(figsize=(7.7,4.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[(.02,.61,'Four street views','0°, 90°, 180°, 270°'),(.36,.61,'Frozen DINOv3 ViT-L/16','4 × 28 × 28 × 1,024'),
        (.70,.61,'Panorama context','Circular ring; 3 × 3 mixing'),(.70,.12,'BatchTopK SAE','4,096 dimensions; k = 32'),
        (.36,.12,'Post-hoc interpretation','Activation evidence + Qwen'),(.02,.12,'City landscape analysis','22 classes + eligible combinations')]
    for i,(x,y,title,sub) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),.27,.26,boxstyle='round,pad=.014',fc=['#dce9e7','#dbe7f0','#e5e5f1','#f2e7d5','#e5ecdc','#e6e9eb'][i],ec='#657572',lw=.8))
        ax.text(x+.135,y+.18,title,ha='center',va='center',fontsize=12.5,weight='bold')
        ax.text(x+.135,y+.075,sub,ha='center',va='center',fontsize=11.5)
    for a,b in [((.30,.74),(.34,.74)),((.64,.74),(.68,.74)),((.835,.59),(.835,.41)),((.68,.25),(.65,.25)),((.34,.25),(.30,.25))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#4a605d'))
    save(fig,'Fig_ViTL30_Workflow')

def panorama_diagram():
    # Reuse a genuine four-view panorama; the diagram expresses the exact mixer.
    manifest=json.loads((INTERNAL/'examples/manifest.json').read_text())['examples']
    src=Image.open(manifest[0]['original']).convert('RGB')
    fig=plt.figure(figsize=(7.5,5.3))
    ax=fig.add_axes([.08,.69,.84,.19]);ax.imshow(src);ax.axis('off')
    fig.text(.5,.94,'Four views at one panorama location',ha='center',fontsize=13,weight='bold')
    fig.text(.5,.64,'Frozen DINOv3-L: four 28 × 28 × 1,024 patch grids',ha='center',fontsize=13)
    ax=fig.add_axes([.08,.34,.84,.21]);ax.set(xlim=(0,112),ylim=(0,28));ax.set_aspect('equal')
    for h,c in enumerate(['#d3e3ed','#dce9df','#eee3ce','#e4dceb']):
        ax.add_patch(plt.Rectangle((h*28,0),28,28,fc=c,ec='#687d7c',lw=1))
        ax.text(h*28+14,14,f'{h*90}°',ha='center',va='center',fontsize=13)
    for x in range(0,113,4):ax.plot([x,x],[0,28],c='white',lw=.5)
    for y in range(0,29,4):ax.plot([0,112],[y,y],c='white',lw=.5)
    ax.axis('off');fig.text(.5,.29,'28 × 112 circular panorama ring; horizontal wrap-around',ha='center',fontsize=13)
    fig.text(.5,.18,r'$\widetilde{x} = \mathrm{normalize}\!\left[0.75x + 0.25\,\mathrm{mean}_{\mathcal{N}_8}(x)\right]$',ha='center',fontsize=14)
    fig.text(.5,.065,'Flatten 3,136 contextualized tokens → BatchTopK SAE',ha='center',fontsize=13,weight='bold')
    save(fig,'Fig_ViTL30_Panorama_Construction')

def simple_examples():
    units=json.loads((PATTERNS/'units.json').read_text())
    patterns={r['pattern_id']:r for r in json.loads((PATTERNS/'patterns.json').read_text())}
    review=json.loads((PATTERNS/'evidence_review.json').read_text())['examples']
    approved={(r['pattern_id'],r['city'],r['panoid']) for r in review if r['status']=='plausible_spot_check'}
    pool=[r for r in json.loads((PATTERNS/'examples.json').read_text()) if (r['pattern_id'],r['city'],r['panoid']) in approved and r['pattern_id'] in patterns]
    selected=[];seen=set();families=set()
    while pool and len(selected)<4:
        r=max(pool,key=lambda r:(len({units[i]['family'] for i in r['units']}-families),len(r['units'])))
        selected.append(r);seen.add(r['pattern_id']);families.update(units[i]['family'] for i in r['units'])
        pool=[r for r in pool if r['pattern_id'] not in seen]
    evidence_panels(selected,units,'Fig_ViTL30_Current_Composition_Examples',ncols=2)

def evidence_panels(rows,units,stem,ncols=2,headings=None):
    from .composition_evidence import COLORS
    nrows=int(np.ceil(len(rows)/ncols))
    fig=plt.figure(figsize=(11.1,nrows*2.8+.4))
    left=.025;gap=.035;w=(.95-gap*(ncols-1))/ncols
    for k,row in enumerate(rows):
        col=k%ncols;rr=k//ncols;x=left+col*(w+gap);step=.91/nrows;top=.945-rr*step
        title=row.get('heading',name(row['city'])+' | '+row['city'].split('/')[0])
        fig.text(x,top,title,fontsize=13,weight='bold')
        ids=row.get('units',json.loads(row.get('unit_ids','[]')))
        labs=row.get('english_labels',[UNIT_NAMES.get(units[i]['label'],units[i]['label']) for i in ids])
        # A separate vector legend per semantic component; retain full panorama.
        for j,(lab,color) in enumerate(zip(labs,COLORS)):
            # Dark text stays legible on paper; colored swatches match the maps.
            yy=top-.045-j*.055*(3/nrows)
            fig.text(x,yy,'■',fontsize=13,color=['#aa7600','#008b99','#b544c9','#659f28'][j])
            fig.text(x+.015,yy,textwrap.fill(lab,16),fontsize=12,va='top',linespacing=1.2)
        height=(step-.05)/2
        for j,field in enumerate(('original','overlay')):
            ax=fig.add_axes([x+w*.30,top-.025-(j+1)*height,w*.70,height*.96])
            im=Image.open(row[field]);ax.imshow(im);ax.axis('off')
    save(fig,stem)

def multicountry():
    examples=[]
    for folder in ('examples','examples_other_cities'):
        examples+=json.loads((INTERNAL/folder/'manifest.json').read_text())['examples']
    chosen=['Bangladesh/Dhaka','Brazil/RiodeJaneiro','UAE/Dubai','Sweden/Stockholm','Singapore/Singapore','US/Chicago']
    units=json.loads((PATTERNS/'units.json').read_text())
    table=pd.read_csv(INTERNAL/'city_patterns.csv')
    rows=[]
    for city in chosen:
        r=next(r.copy() for r in examples if r['city']==city and r['rank']==1)
        row=table[(table.city==city)&(table.pattern_id==r['pattern_id'])].iloc[0]
        r['units']=json.loads(row.unit_ids);rows.append(r)
    evidence_panels(rows,units,'Fig_ViTL30_Multicountry_Composition_Examples',ncols=2)
    (OUT/'multicountry_evidence.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))

def appendix_atlas(cells):
    from matplotlib.backends.backend_pdf import PdfPages
    means=cells.groupby('city')[list(CLASSES)].mean()
    rates=pd.read_csv(PATTERNS/'city_distribution.csv')
    key=pd.read_csv(OLD/'all_composition_key.csv').set_index('pattern_id')
    target=OUT/'cities';target.mkdir(exist_ok=True)
    with PdfPages(OUT/'Fig_ViTL30_City_Composition_Atlas.pdf') as atlas:
        for city,shares in means.iterrows():
            fig=plt.figure(figsize=(6.8,9.0))
            a=fig.add_axes([.37,.49,.60,.43]);b=fig.add_axes([.11,.07,.86,.34])
            a.barh(range(22),shares.to_numpy()*100,color=PALETTE,height=.72)
            a.set_yticks(range(22),[LABELS[c] for c in CLASSES],fontsize=11.5)
            a.invert_yaxis();a.set(xlim=(0,100),xlabel='Share of positive SAE response (%)')
            a.set_title('a  All 22 semantic classes',loc='left',fontsize=12.5,weight='bold')
            a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
            top=rates[rates.city==city].nlargest(8,'cell_weighted_rate')
            y=np.arange(8)*2
            b.barh(y+.65,top.cell_weighted_rate*100,color='#327f9d',height=.5)
            for yy,row in zip(y,top.itertuples()):
                text=f'{key.loc[row.pattern_id,"atlas_id"]}  {key.loc[row.pattern_id,"english_units"]}'
                b.text(0,yy-.03,textwrap.fill(text,48),fontsize=11.5,va='center',linespacing=1.05)
                b.text(row.cell_weighted_rate*100+.6,yy+.65,f'{row.cell_weighted_rate*100:.1f}',fontsize=11,va='center')
            b.set(ylim=(15.5,-.7),xlim=(0,105),xlabel='Cell-balanced matched panoramas (%)')
            b.set_yticks([]);b.set_title('b  Eight most prevalent combinations',loc='left',fontsize=12.5,weight='bold')
            b.grid(axis='x',alpha=.15);b.set_axisbelow(True)
            fig.text(.04,.968,name(city),fontsize=14,weight='bold')
            g=cells[cells.city==city]
            fig.text(.04,.943,f'{int(g.sampled_panoramas.sum()):,} panoramas; {len(g):,} observed 1-km cells',fontsize=12)
            slug=city.replace('/','__').replace('Brasília','Brasilia');stem=f'City_{slug}_Composition'
            atlas.savefig(fig,bbox_inches='tight',pad_inches=.08)
            fig.savefig(target/f'{stem}.pdf',bbox_inches='tight',pad_inches=.08)
            fig.savefig(target/f'{stem}.png',dpi=160,bbox_inches='tight',pad_inches=.08)
            plt.close(fig)
            shutil.copy2(target/f'{stem}.pdf',ASSETS/'cities'/f'{stem}.pdf')
    print('30 appendix city pages regenerated',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['plots','umap'],default='plots');a=p.parse_args();style()
    if a.phase=='umap':umap_plot();return
    cells=pd.read_parquet(STAGE/'cell_class_shares.parquet')
    city_plots(cells);pattern_plots();study_map();workflow();panorama_diagram();simple_examples();multicountry();appendix_atlas(cells)
    print('Cached-data figure revisions complete',flush=True)
if __name__=='__main__':main()
