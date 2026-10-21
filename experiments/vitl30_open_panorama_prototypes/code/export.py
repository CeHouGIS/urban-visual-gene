"""Development-only four-view evidence; masks show reconstruction attribution."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import torch
from .data import OUT, digest, write_json
from .model import RingDictionary

def render_four(row, contribution, destination):
    """Preserve each native aspect; heat/mask coordinates map to its full view."""
    originals, overlays, masks, hard_masks = [], [], [], []
    for h,heading in enumerate((0,90,180,270)):
        source = Path(row[f'image_{heading}'])
        if source.stem.rsplit('_',1)!=[str(row.panoid),str(heading)]:
            raise ValueError('Panorama identity or heading differs from image filename')
        with Image.open(source) as image:
            image = ImageOps.contain(image.convert('RGB'),(320,256),Image.Resampling.LANCZOS)
        heat = contribution[:,h*14:(h+1)*14]
        # Contribution share is comparable within/across views and examples (0..1).
        strength = np.asarray(Image.fromarray(heat.astype(np.float32)).resize(image.size,Image.Resampling.BILINEAR))
        hard = np.asarray(Image.fromarray((heat>=.35).astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST))>0
        rgb = np.asarray(image).astype(np.float32)
        alpha = (.7*strength)[...,None]
        color = np.array([255,65,15],np.float32)
        overlay = Image.fromarray(np.uint8(np.clip(rgb*(1-alpha)+color*alpha,0,255)))
        masked = Image.fromarray(np.uint8(rgb*np.where(hard,1.,.1)[...,None]))
        originals.append(image); overlays.append(overlay); masks.append(masked)
        hard_masks.append((heat>=.35).astype(int).tolist())
    tile,bar = 320,23
    height = max(image.height for image in originals)
    canvas = Image.new('RGB',(4*tile,3*(height+bar)),'#12221e')
    draw = ImageDraw.Draw(canvas)
    for level,views in enumerate((originals,overlays,masks)):
        for h,image in enumerate(views):
            x,y = h*tile+(tile-image.width)//2,level*(height+bar)+bar+(height-image.height)//2
            canvas.paste(image,(x,y))
            label = ('original','contribution share 0..1','mask share >=0.35')[level]
            draw.text((h*tile+8,level*(height+bar)+5),f'{h*90} deg | {label}',fill='white')
    destination.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(destination,'WEBP',quality=87,method=4)
    return hard_masks

@torch.inference_mode()
def export(out=OUT,device='cuda:1',per_prototype=6):
    out = Path(out)
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU inference disabled')
    torch.set_num_threads(4)
    saved = torch.load(out/'best.pt',map_location=device,weights_only=False)
    summary = json.loads((out/'training_summary.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    if saved['identity']!=summary['identity'] or saved['identity']!=protocol['identity']:
        raise ValueError('Evidence/model/protocol identity mismatch')
    model = RingDictionary(saved['templates']).to(device)
    frame = pd.read_parquet(out/'panoramas.parquet')
    inputs = np.load(out/'inputs.npy',mmap_mode='r')
    dev = frame.index[frame.prototype_split=='dev'].to_numpy()
    weights = np.zeros((len(dev),len(saved['ids'])),np.float32)
    errors = np.zeros(len(dev),np.float32)
    for start in range(0,len(dev),256):
        x = torch.from_numpy(np.array(inputs[dev[start:start+256]])).to(device).float()
        reconstruction,code = model(x)
        weights[start:start+len(x)] = code.reshape(len(x),-1,4).sum(2).cpu().numpy()
        errors[start:start+len(x)] = (x-reconstruction).square().flatten(1).sum(1).cpu().numpy()
    np.save(out/'dev_template_weights.npy',weights)
    np.save(out/'dev_nmse.npy',errors)
    hierarchy = json.loads((Path(__file__).resolve().parents[3]/'experiments/vitl30_relational_prototypes/results/hierarchy.json').read_text())
    feature_groups = np.load(out/'dimension_groups.npy')
    by_feature = {row['feature']:row for row in hierarchy['dimensions']}
    prototypes = []
    gallery = out/'gallery'
    gallery.mkdir(exist_ok=True)
    for j,stable_id in enumerate(saved['ids']):
        order = np.argsort(-weights[:,j],kind='stable')
        selected,seen,blocks = [],set(),set()
        for distinct in (True,False):
            for local in order:
                if weights[local,j]<=.02:
                    break
                i = int(dev[local]); row = frame.iloc[i]
                if i in selected or row.spatial_block in blocks or (distinct and row.city in seen):
                    continue
                selected.append(i); seen.add(row.city); blocks.add(row.spatial_block)
                if len(selected)>=per_prototype:
                    break
            if len(selected)>=per_prototype:
                break
        examples = []
        for rank,i in enumerate(selected):
            row = frame.iloc[i]
            if row.prototype_split!='dev':
                raise ValueError('Only development images may be exported')
            x = torch.from_numpy(np.array(inputs[i:i+1])).to(device).float()
            reconstruction,code = model(x)
            rotated = torch.stack([torch.roll(model.templates[j],r*14,-1) for r in range(4)])
            contribution = (rotated*code.reshape(1,-1,4)[0,j,:,None,None,None]).sum(0)
            # Allocate each observed input coordinate in proportion to predicted contribution.
            allocated = x[0]*contribution/reconstruction[0].clamp_min(1e-8)
            share = allocated.sum(0)/x[0].sum(0).clamp_min(1e-8)
            share = share.clamp(0,1).cpu().numpy()
            filename = f'p{stable_id:04d}_example{rank:02d}.webp'
            masks = render_four(row,share,gallery/filename)
            examples.append({'image':filename,'city':row.city,'panoid':row.panoid,'partition':'dev',
                'lat':float(row.lat),'lon':float(row.lon),'spatial_block':row.spatial_block,
                'coefficient':float(code.reshape(1,-1,4)[0,j].sum()),
                'nmse':float((x-reconstruction).square().sum()),
                'rotation_coefficients':code.reshape(1,-1,4)[0,j].cpu().tolist(),
                'mask_fraction':float((share>=.35).mean()),'mask_grid_by_heading':masks,
                'source_images':{str(h):str(row[f'image_{h}']) for h in (0,90,180,270)}})
        group_weight = model.templates[j].sum((1,2)).cpu().numpy()
        leaders = group_weight.argsort()[::-1][:8]
        group_descriptions = []
        for group in leaders:
            labels = Counter(by_feature[int(k)].get('family_zh','未知') for k in np.where(feature_groups==group)[0])
            group_descriptions.append({'decoder_group':int(group),'weight_share':float(group_weight[group]/group_weight.sum()),
                'families':labels.most_common(3),'meaning_status':'mixed decoder group; labels are posthoc and unverified'})
        prototypes.append({'id':stable_id,'train_support':summary['train']['support'][j],
            'train_cities':summary['train']['support_cities'][j],'dev_support':int((weights[:,j]>.02).sum()),
            'top_groups':group_descriptions,'examples':examples,'semantic_status':'candidate_pending_review'})
    report = {'identity':saved['identity'],'checkpoint_sha256':protocol['checkpoint_sha256'],
        'prototype_checkpoint_sha256':digest(out/'best.pt'),'best_epoch':saved['epoch'],
        'headings':[0,90,180,270],'native_grid':[28,112],'model_grid':[14,56],
        'full_panorama':True,'test_used':False,'projection':'four separate perspective images in heading order',
        'mask_definition':'share of observed grouped activation attributed to this template, threshold 0.35; not object segmentation',
        'semantic_validation':'pending','prototypes':prototypes,'training_summary':summary}
    write_json(gallery/'index.json',report)
    return {'prototypes':len(prototypes),'examples':sum(len(p['examples']) for p in prototypes),
            'path':str(gallery),'test_used':False}
