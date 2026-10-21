"""GPU-only whole-ring fits with recorded residual births/merges/retirement."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .data import OUT, digest, status, write_json
from .model import RingDictionary, best_rotation_similarity, rotated_bank, fit_loss

def save_checkpoint(path, obj):
    tmp = path.with_suffix('.partial.pt')
    torch.save(obj, tmp)
    tmp.replace(path)

def panel(frame, split, per_city):
    selected = []
    for _, rows in frame[frame.prototype_split == split].groupby('city', sort=True):
        selected.extend(rows.sample(min(per_city,len(rows)),random_state=42).index.tolist())
    return np.array(selected,np.int64)

@torch.inference_mode()
def score(model,data,ids,cities,batch=256,support_threshold=.02):
    losses,zero,covered,valid,full,uses = [],0,0,0,0,[]
    supports = torch.zeros(len(model.templates),device=data.device,dtype=torch.long)
    city_support = torch.zeros(len(model.templates),int(cities.max())+1,device=data.device,dtype=torch.bool)
    for at in range(0,len(ids),batch):
        rows = ids[at:at+batch]
        x = data[rows].float()
        reconstruction,code = model(x)
        difference = (x-reconstruction).square()
        losses.extend(difference.flatten(1).sum(1).cpu().tolist())
        zero += int(x.flatten(1).norm(dim=1).lt(1e-8).sum())
        energy = x.square().sum(1)
        good = (difference.sum(1)<=.5*energy)&(energy>1e-12)
        covered += int(good.sum()); valid += int(energy.gt(1e-12).sum())
        per_direction = good.reshape(len(x),14,4,14).permute(0,2,1,3).float().mean((2,3))
        full += int((per_direction>=.8).all(1).sum())
        active = code.reshape(len(x),-1,4).sum(2)>support_threshold
        supports += active.sum(0)
        uses.extend(active.sum(1).cpu().tolist())
        for city in cities[rows].unique():
            city_support[:,city] |= active[cities[rows]==city].any(0)
    return {'panoramas':len(ids),'nmse_unit_normalized':float(np.mean(losses)),
        'zero_input_panoramas':zero,'zero_reconstruction_baseline':1.0,
        'mean_templates_per_panorama':float(np.mean(uses)),
        'group_pooled_patch_coverage_all_denominator':covered/(len(ids)*784),
        'group_pooled_patch_coverage_positive_denominator':covered/max(valid,1),
        'four_directions_each_at_least_80pct':full/len(ids),
        'coverage_definition':'64 grouped channels,14x56 pooled sites; site NMSE<=0.5, not original4096 or semantic coverage',
        'support':supports.cpu().tolist(),'support_cities':city_support.sum(1).cpu().tolist()}

@torch.inference_mode()
def residual_candidates(model,data,ids,cities,max_candidates=8):
    x = data[ids].float()
    reconstruction,_ = model(x)
    residual = (x-reconstruction).clamp_min(0)
    norm = residual.flatten(1).norm(dim=1)
    unit = residual/norm.clamp_min(1e-8)[:,None,None,None]
    # Calibrate spatial recurrence against training-only shuffled residual maps.
    # Use one permutation per panorama across channels: preserve composition and
    # channel coactivation while destroying original spatial arrangement.
    rng = torch.Generator(device=data.device).manual_seed(20261016)
    shuffled = unit.flatten(2).clone()
    for i in range(len(shuffled)):
        permutation = torch.randperm(shuffled.shape[-1],generator=rng,device=data.device)
        shuffled[i] = shuffled[i,:,permutation]
    null_scores = []
    for seed in norm.argsort(descending=True)[:32].tolist():
        null_scores.append((shuffled.flatten(1)@rotated_bank(unit[seed:seed+1]).T).max(1).values)
    null_threshold = float(torch.quantile(torch.cat(null_scores),.995))
    threshold = max(.30,null_threshold)
    proposals,used = [],torch.zeros(len(ids),device=data.device,dtype=torch.bool)
    for seed in norm.argsort(descending=True)[:96].tolist():
        if used[seed] or norm[seed]<.05:
            continue
        sim,rotations = (unit.flatten(1)@rotated_bank(unit[seed:seed+1]).T).max(1)
        members = (sim>=threshold)&~used
        member_ids = members.nonzero().flatten()
        if len(member_ids)<6 or len(cities[ids[member_ids]].unique())<3:
            continue
        aligned = torch.stack([torch.roll(unit[j],-int(rotations[j])*14,-1) for j in member_ids.tolist()])
        candidate = aligned.mean(0)
        candidate /= candidate.norm().clamp_min(1e-8)
        proposals.append((candidate,{'panel_support':len(member_ids),
            'panel_cities':len(cities[ids[member_ids]].unique()),'residual_similarity':threshold,
            'spatial_shuffle_null_quantile_995':null_threshold}))
        used |= members
        if len(proposals)>=max_candidates:
            break
    return proposals

def train(out=OUT,device='cuda:1',epochs=12,capacity=96,seed=42,batch=256):
    out = Path(out)
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU training is disabled')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    protocol = json.loads((out/'protocol.json').read_text())
    cfg = {'epochs':epochs,'capacity_resource_limit':capacity,'seed':seed,'batch':batch,
        'initial_atoms':4,'birth_every':2,'birth_max_per_event':8,
        'birth_min_panel_panos':6,'birth_min_cities':3,'merge_cosine':.97,
        'birth_similarity':'max(0.30, train-only within-panorama spatial-shuffle 99.5 percentile)',
        'null_scope':'threshold calibration, not an independent significance test',
        'support_coefficient_threshold':.02,'retire_min_train_panos':20,
        'retire_min_train_cities':3,'retirement_warmup_epochs':4,
        'nnls':{'screened_atoms_rotations':8,'steps':32,'l1':.015},
        'learning_rate':.0002,'model_code_sha256':digest(Path(__file__).with_name('model.py')),
        'train_code_sha256':digest(__file__),'scope':'exploratory dev-selected open dictionary'}
    if (out/'train_config.json').exists() and json.loads((out/'train_config.json').read_text())!=cfg:
        raise ValueError('Training configuration changed; use another output directory')
    write_json(out/'train_config.json',cfg)
    frame = pd.read_parquet(out/'panoramas.parquet')
    if not frame.prototype_split.isin(['train','dev']).all():
        raise ValueError('Test data cannot enter training')
    status(out,'training',stage='loading_ring_inputs_gpu',device=device,epoch=0,epochs=epochs,
           detail=f'正在加载{len(frame):,}个训练/开发全景；测试封存。')
    data = torch.from_numpy(np.load(out/'inputs.npy')).to(device)
    _,city_codes = np.unique(frame.city,return_inverse=True)
    cities = torch.tensor(city_codes,device=device)
    train_ids = torch.tensor(frame.index[frame.prototype_split=='train'].to_numpy(),device=device)
    dev_ids = torch.tensor(frame.index[frame.prototype_split=='dev'].to_numpy(),device=device)
    generation_ids = torch.tensor(panel(frame,'train',32),device=device)
    screening_ids = torch.tensor(panel(frame,'dev',8),device=device)
    history,events,ids,born,next_id,best,start = [],[],list(range(4)),{i:0 for i in range(4)},4,float('inf'),0
    generator = torch.Generator(device=device).manual_seed(seed)
    chosen = train_ids[torch.randperm(len(train_ids),generator=generator,device=device)[:4]]
    model = RingDictionary(data[chosen].float()).to(device)
    optimizer = torch.optim.Adam([model.templates],lr=cfg['learning_rate'])
    if (out/'last.pt').exists():
        saved = torch.load(out/'last.pt',map_location=device,weights_only=False)
        if saved['identity']!=protocol['identity'] or saved['config']!=cfg:
            raise ValueError('Checkpoint identity differs')
        model = RingDictionary(saved['templates']).to(device)
        optimizer = torch.optim.Adam([model.templates],lr=cfg['learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        history,events,ids,born,next_id,best,start = [saved[k] for k in ('history','events','ids','born','next_id','best','epoch')]
        generator.set_state(saved['rng'].cpu())
    else:
        events.append({'epoch':0,'event':'initialize','ids':ids.copy(),'source':'training panoramas only'})
    started = time.monotonic()
    def replace(templates,new_ids):
        nonlocal model,optimizer,ids
        model = RingDictionary(templates).to(device)
        optimizer = torch.optim.Adam([model.templates],lr=cfg['learning_rate'])
        ids = new_ids
    for epoch in range(start+1,epochs+1):
        order = train_ids[torch.randperm(len(train_ids),generator=generator,device=device)]
        total = 0.
        for at in range(0,len(order),batch):
            x = data[order[at:at+batch]].float()
            optimizer.zero_grad(set_to_none=True)
            reconstruction,_ = model(x)
            loss = fit_loss(x,reconstruction)
            loss.backward(); optimizer.step(); model.project()
            total += float(loss.detach())*len(x)
            if at%(batch*20)==0:
                status(out,'training',stage='dictionary_fit',device=device,epoch=epoch,epochs=epochs,
                    completed=at,total=len(order),templates=len(ids),
                    detail=f'完整全景训练第{epoch}/{epochs}轮，{len(ids)}个候选模板；{at:,}/{len(order):,}训练全景。')
        train_score = score(model,data,train_ids,cities,batch)
        if epoch%2==0:
            similarity,rotation = best_rotation_similarity(model.templates)
            pairs = torch.triu(similarity,diagonal=1)
            while len(ids)>1 and float(pairs.max())>=.97:
                i,j = divmod(int(pairs.argmax()),len(ids))
                templates = model.templates.detach().clone()
                templates[i] += torch.roll(templates[j],int(rotation[i,j])*14,-1)
                keep = [k for k in range(len(ids)) if k!=j]
                events.append({'epoch':epoch,'event':'merge','kept':ids[i],'removed':ids[j],
                    'cosine':float(similarity[i,j]),'quarter_turn':int(rotation[i,j])})
                replace(templates[keep],[ids[k] for k in keep])
                similarity,rotation = best_rotation_similarity(model.templates)
                pairs = torch.triu(similarity,diagonal=1)
            train_score = score(model,data,train_ids,cities,batch)
            if epoch>=4:
                keep = [j for j,k in enumerate(ids) if epoch-born[k]<2 or
                    (train_score['support'][j]>=20 and train_score['support_cities'][j]>=3)]
                if keep and len(keep)<len(ids):
                    events.append({'epoch':epoch,'event':'retire','ids':[k for j,k in enumerate(ids) if j not in keep]})
                    replace(model.templates.detach()[keep],[ids[j] for j in keep])
            base = score(model,data,screening_ids,cities,batch)['nmse_unit_normalized']
            for candidate,evidence in residual_candidates(model,data,generation_ids,cities):
                if len(ids)>=capacity:
                    events.append({'epoch':epoch,'event':'resource_capacity_reached','capacity':capacity})
                    break
                proposed = RingDictionary(torch.cat([model.templates.detach(),candidate[None]])).to(device)
                trial = score(proposed,data,screening_ids,cities,batch)['nmse_unit_normalized']
                if trial<base-1e-5:
                    events.append({'epoch':epoch,'event':'birth','id':next_id,'dev_screen_before':base,
                        'dev_screen_after':trial,**evidence})
                    born[next_id] = epoch
                    replace(proposed.templates.detach(),ids+[next_id])
                    next_id += 1; base = trial
            train_score = score(model,data,train_ids,cities,batch)
        dev_score = score(model,data,dev_ids,cities,batch)
        record = {'epoch':epoch,'templates':len(ids),'train_optimization_nmse':total/len(train_ids),
            'train':train_score,'dev':dev_score,'elapsed_seconds':time.monotonic()-started}
        history.append(record)
        improved = dev_score['nmse_unit_normalized']<best
        best = min(best,dev_score['nmse_unit_normalized'])
        checkpoint = {'identity':protocol['identity'],'config':cfg,'templates':model.templates.detach().cpu(),
            'ids':ids,'born':born,'next_id':next_id,'best':best,'epoch':epoch,'history':history,
            'events':events,'optimizer':optimizer.state_dict(),'rng':generator.get_state()}
        save_checkpoint(out/'last.pt',checkpoint)
        if improved:
            save_checkpoint(out/'best.pt',checkpoint)
        write_json(out/'history.json',history); write_json(out/'events.json',events)
        status(out,'training',stage='epoch_complete',device=device,epoch=epoch,epochs=epochs,
            templates=len(ids),metrics=dev_score,elapsed_seconds=time.monotonic()-started,
            detail=f'第{epoch}/{epochs}轮已完成；{len(ids)}个候选，开发集组级NMSE={best:.4f}。')
        print('EPOCH',epoch,'templates',len(ids),'dev_nmse',dev_score['nmse_unit_normalized'],flush=True)
    saved = torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    result = {'identity':protocol['identity'],'status':'complete','best_epoch':saved['epoch'],
        'templates':len(saved['ids']),'ids':saved['ids'],'dev':saved['history'][-1]['dev'],
        'train':saved['history'][-1]['train'],'epochs':epochs,'events':events,'config':cfg,
        'test_sealed':True,'semantic_validation':'pending','full4096_coverage':'not_evaluated',
        'warning':'exploratory dev-selected grouped pooled reconstruction; not semantic coverage',
        'checkpoint_sha256':protocol['checkpoint_sha256']}
    write_json(out/'training_summary.json',result)
    status(out,'training',stage='exporting_dev_evidence',templates=result['templates'],metrics=result['dev'],
        detail='训练已完成，正在导出完整四方向实例和贡献mask；语义核查待开展。')
    return result
