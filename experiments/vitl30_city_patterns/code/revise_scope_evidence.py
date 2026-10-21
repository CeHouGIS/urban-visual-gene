"""Nine panorama examples with sample-defined geographical recurrence scopes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModel
from .revise_chapter5_figures import OUT, INTERNAL, PATTERNS, FOLLOWUP, style, save, name, UNIT_NAMES
from .render_city_distinctive_examples import choose_indices
from .composition_evidence import infer, render, COLORS
from .extract import image_paths, EXPECTED_CHECKPOINT
from sae_experiments.panorama_vitl30 import MODEL_DIR, OUTPUT as SAE_OUTPUT
from sae_experiments.panorama_vitl30_analysis import model_and_identity

SELECTION = [
 ('country','cd8d335c382b','Bangladesh/Dhaka','Bangladesh'),
 ('regional','c10ad95faeb7','Italy/Naples','Europe: Austria + Italy'),
 ('shared','3126a722d8dd','Singapore/Singapore','21 of 30 cities'),
 ('country','867d8c3032f1','UAE/Dubai','United Arab Emirates'),
 ('regional','6c4c7543befd','Indonesia/Yogyakarta','South / SE Asia: BD + ID'),
 ('shared','a4980b339213','Mexico/Monterrey','20 of 30 cities'),
 ('country','a9e43f224ca2','Sweden/Stockholm','Sweden'),
 ('regional','043a1825f49a','Brazil/SaoPaulo','Latin America: AR + BR + MX'),
 ('shared','ae5f893e3d29','US/Chicago','20 of 30 cities'),
]

def run():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU inference')
    style();torch.set_num_threads(4)
    dest=OUT/'scope_evidence';dest.mkdir(exist_ok=True)
    units=json.loads((PATTERNS/'units.json').read_text())
    rows=pd.read_csv(INTERNAL/'city_patterns.csv')
    support=pd.read_csv(INTERNAL/'pattern_city_support.csv').set_index('pattern_id')
    frame=pd.read_parquet(FOLLOWUP/'stages/candidate_combinations/panorama_elements.parquet',columns=['city','panoid','grid_x','grid_y'])
    assignment=np.load(PATTERNS/'assignment.npy',mmap_mode='r');scores=np.load(PATTERNS/'scores.npy',mmap_mode='r')
    thresholds=np.asarray(json.loads((PATTERNS/'calibration.json').read_text())['unit_thresholds'])
    sae,checkpoint=model_and_identity(SAE_OUTPUT,'cuda:0')
    if checkpoint!=EXPECTED_CHECKPOINT:raise ValueError('Wrong frozen checkpoint')
    dino=AutoModel.from_pretrained(MODEL_DIR,local_files_only=True).eval().requires_grad_(False).to(device='cuda:0',dtype=torch.bfloat16)
    manifest=[]
    for scope,pid,city,heading in SELECTION:
        row=rows[(rows.city==city)&(rows.pattern_id==pid)]
        if len(row)!=1 or not bool(row.iloc[0].replicated):raise ValueError('Not a recurrent city pattern')
        ids=json.loads(row.iloc[0].unit_ids)
        cities=support.loc[pid].city_names.split(' | ');countries=sorted({c.split('/')[0] for c in cities})
        if scope=='country' and len(countries)!=1:raise ValueError('Wrong country scope')
        if scope=='regional' and len(countries)<2:raise ValueError('Regional needs multiple countries')
        if scope=='shared' and len(cities)<=15:raise ValueError('Shared needs most study cities')
        for rank,(pos,strength) in enumerate(choose_indices(frame,assignment,scores,thresholds,units,city,ids,2),1):
            sample=frame.iloc[pos];target=dest/f'{pid}_{rank}';target.mkdir(exist_ok=True)
            im=[]
            for path in image_paths(city,sample.panoid):
                with Image.open(path) as image:im.append(image.convert('RGB'))
            marker={'checkpoint':checkpoint,'city':city,'panoid':sample.panoid,'units':ids}
            cache=target/'activation.npz'
            if cache.exists():
                data=np.load(cache)
                if json.loads(str(data['identity']))!=marker:raise ValueError('Attribution cache mismatch')
                maps=data['maps']
            else:
                maps=infer(im,dino,sae,ids,units,'cuda:0');np.savez_compressed(cache,identity=json.dumps(marker),maps=maps)
            if not np.isfinite(maps).all() or not (maps.sum(axis=(1,2,3))>0).all():raise ValueError('Invalid evidence')
            render(im,maps,target)
            manifest.append({'scope':scope,'pattern_id':pid,'city':city,'heading':heading,
                'units':ids,'english_labels':[UNIT_NAMES[units[i]['label']] for i in ids],
                'panoid':sample.panoid,'rank':rank,'minimum_unit_score_ratio':strength,
                'cities_supported':cities,'countries_supported':countries,
                'original':str(target/'original.webp'),'overlay':str(target/'overlay.webp')})
            print(f'{pid} {city}: {rank}/2',flush=True)
    (dest/'manifest.json').write_text(json.dumps({'examples':manifest,'scope_rule':'Passes existing within-city discovery and spatial holdout cutoffs; not evidence of absence elsewhere',
        'checkpoint':checkpoint},indent=2))
    plot(manifest)

def plot(manifest):
    import matplotlib.pyplot as plt
    style();fig=plt.figure(figsize=(11.5,9.2))
    columns=['Country-limited support','Regional support','Widely shared support']
    for col,title in enumerate(columns):fig.text(.025+col*.33,.972,title,fontsize=15,weight='bold')
    for k,(scope,pid,city,heading) in enumerate(SELECTION):
        r=next(r for r in manifest if r['pattern_id']==pid and r['rank']==1)
        col=k%3;rr=k//3;x=.025+col*.33;top=.93-rr*.30
        fig.text(x,top,heading,fontsize=13.5,weight='bold')
        city_count=len(r['cities_supported']);country_count=len(r['countries_supported'])
        city_word='city' if city_count==1 else 'cities'
        country_word='country' if country_count==1 else 'countries'
        fig.text(x,top-.022,f'{name(city)} | {city_count} {city_word} / {country_count} {country_word}',fontsize=12.5)
        for i,(lab,color) in enumerate(zip(r['english_labels'],COLORS)):
            fig.text(x,top-.047-i*.018,'■ '+lab,fontsize=12.5,color='#a57500' if i==0 else ['','#008fa0','#b34cc9','#5d9b26'][i])
        for j,field in enumerate(('original','overlay')):
            ax=fig.add_axes([x,top-.165-j*.072,.31,.064]);ax.imshow(Image.open(r[field]));ax.axis('off')
    save(fig,'Fig_ViTL30_Geographic_Scope_Examples')

if __name__=='__main__':run()
