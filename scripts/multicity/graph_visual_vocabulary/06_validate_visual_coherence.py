#!/usr/bin/env python3
"""Create panorama evidence and independently validate communities with DINO crops.

The ``all`` command is a subprocess orchestrator so torch inference and scipy
statistics never share one process (see the repository OpenMP crash report).
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, l2_rows, load_config, save_json


class TarReader:
    def __init__(self, limit: int = 16):
        self.limit=limit; self.fds: OrderedDict[str,int]=OrderedDict()
    def image(self,path: str,offset: int,size: int) -> Image.Image:
        if path in self.fds:
            fd=self.fds.pop(path); self.fds[path]=fd
        else:
            fd=os.open(path,os.O_RDONLY); self.fds[path]=fd
            if len(self.fds)>self.limit: _,old=self.fds.popitem(last=False); os.close(old)
        payload=os.pread(fd,size,offset)
        if len(payload)!=size: raise OSError(f"short TAR read {path}:{offset}")
        with Image.open(io.BytesIO(payload)) as image: return image.convert("RGB")
    def close(self):
        for fd in self.fds.values(): os.close(fd)
        self.fds.clear()


def load_city_manifest(root: Path, city_slug: str) -> pd.DataFrame:
    return pd.read_parquet(root/"manifests"/f"{city_slug}.parquet").sort_values(["pano_index","direction_index"]).reset_index(drop=True)


def exemplar_patch_table(config: dict, force: bool=False) -> pd.DataFrame:
    cache=config["paths"]["cache_root"]; panorama=config["paths"]["panorama_root"]
    target=cache/"validation/dimension_crop_manifest.csv"
    if target.exists() and not force: return pd.read_csv(target)
    exemplars=pd.read_csv(cache/"views/dimension_top_panoramas.csv"); rows=[]
    for city_index,(city_slug,part) in enumerate(exemplars.groupby("city_slug"),1):
        latent=np.load(panorama/"predictions"/city_slug/"full/panorama_feature_mae_latent.f16.npy",mmap_mode="r")
        for row in part.itertuples(index=False):
            activation=np.asarray(latent[row.pano_index,:,:,row.dimension_id],dtype=np.float32)
            patch=int(np.argmax(activation)); patch_row,patch_column=divmod(patch,56); direction=patch_column//14; local_column=patch_column%14
            rows.append({**row._asdict(),"patch_row":patch_row,"patch_column":patch_column,"direction_index":direction,"local_patch_column":local_column,"max_patch_activation":float(activation[patch_row,patch_column])})
        print(f"patch maxima [{city_index:02d}] {city_slug}: {len(part)}",flush=True)
    frame=pd.DataFrame(rows).sort_values(["dimension_id","rank"]); frame.to_csv(target,index=False); return frame


def source_images(manifest: pd.DataFrame,pano_index: int,reader: TarReader) -> list[Image.Image]:
    group=manifest.loc[manifest.pano_index.eq(pano_index)].sort_values("direction_index")
    if len(group)!=4: raise ValueError(f"pano {pano_index} does not have four directions")
    return [reader.image(str(x.tar_path),int(x.jpg_offset),int(x.jpg_size)) for x in group.itertuples(index=False)]


def crop_from_direction(image: Image.Image,row: int,column: int,size: int=224) -> Image.Image:
    cx=(column+.5)*image.width/14; cy=(row+.5)*image.height/14
    left=max(0,min(image.width-size,int(cx-size/2))); top=max(0,min(image.height-size,int(cy-size/2)))
    return image.crop((left,top,left+size,top+size))


def overlay_panorama(panorama: Image.Image,activation: np.ndarray) -> Image.Image:
    from matplotlib import colormaps
    values=np.asarray(activation,dtype=np.float32); lo=float(np.quantile(values,.05)); hi=float(np.quantile(values,.995)); scaled=np.clip((values-lo)/max(hi-lo,1e-6),0,1)
    rgba=(colormaps["inferno"](scaled)*255).astype(np.uint8); heat=Image.fromarray(rgba[:,:,:3]).resize(panorama.size,Image.Resampling.BILINEAR)
    alpha=Image.fromarray((scaled*180).astype(np.uint8)).resize(panorama.size,Image.Resampling.BILINEAR)
    return Image.composite(heat,panorama,alpha)


def prepare(config: dict,force: bool=False) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    cache=config["paths"]["cache_root"]; data=config["paths"]["paper_data_root"]; figures=config["paths"]["paper_figure_root"]; panorama=config["paths"]["panorama_root"]
    crop_frame=exemplar_patch_table(config,force)
    membership=pd.read_csv(data/"leiden/latent_to_fine.csv"); community_metrics=pd.read_csv(data/"validation/community_graph_metrics.csv"); graph_edges=pd.read_csv(data/"graph/visual_graph_edges.csv")
    fused=np.zeros((512,512),dtype=np.float32); fused[graph_edges.source,graph_edges.target]=graph_edges.weight_fused; fused+=fused.T
    manifests={}; reader=TarReader(); evidence_rows=[]
    try:
        for number,community in enumerate(community_metrics.itertuples(index=False),1):
            members=membership.loc[membership.fine_community.eq(community.community_id),"dimension_id"].to_numpy(int); medoid=int(community.medoid_dimension_id)
            neighbours=[x for x in members[np.argsort(-fused[medoid,members])] if x!=medoid][:int(config["visual_validation"]["neighbour_dimensions"])]
            dimensions=[medoid,*neighbours]; selected=[]; seen=set()
            for dim_index,dimension in enumerate(dimensions):
                limit=10 if dim_index==0 else 3
                for row in crop_frame.loc[crop_frame.dimension_id.eq(dimension)].sort_values("rank").itertuples(index=False):
                    key=(row.city_slug,int(row.pano_index))
                    if key in seen: continue
                    seen.add(key); selected.append(row)
                    if sum(int(x.dimension_id)==dimension for x in selected)>=limit: break
                    if len(selected)>=int(config["visual_validation"]["max_crops_per_community"]): break
                if len(selected)>=int(config["visual_validation"]["max_crops_per_community"]): break
            cell_w,cell_h,columns=840,230,2; sheet=Image.new("RGB",(cell_w*columns,cell_h*math.ceil(len(selected)/columns)),"white")
            for position,row in enumerate(selected):
                if row.city_slug not in manifests: manifests[row.city_slug]=load_city_manifest(panorama,row.city_slug)
                images=source_images(manifests[row.city_slug],int(row.pano_index),reader); pano=Image.new("RGB",(2560,640))
                for index,image in enumerate(images): pano.paste(image,(index*640,0))
                latent=np.load(panorama/"predictions"/row.city_slug/"full/panorama_feature_mae_latent.f16.npy",mmap_mode="r")
                activation=np.asarray(latent[int(row.pano_index),:,:,int(row.dimension_id)],dtype=np.float32); heat=overlay_panorama(pano,activation)
                crop=crop_from_direction(images[int(row.direction_index)],int(row.patch_row),int(row.local_patch_column))
                pane_w=cell_w//3; parts=[pano,heat,crop]
                tile=Image.new("RGB",(cell_w,cell_h),"white")
                for pane,image in enumerate(parts): image.thumbnail((pane_w,cell_h-28),Image.Resampling.LANCZOS); tile.paste(image,(pane*pane_w+(pane_w-image.width)//2,28))
                draw=ImageDraw.Draw(tile); draw.rectangle((0,0,cell_w,27),fill="black"); draw.text((6,7),f"{community.community_id} D{int(row.dimension_id):03d}  ORIGINAL PANORAMA",fill="white"); draw.text((pane_w+6,7),"ACTIVATION",fill="white"); draw.text((2*pane_w+6,7),"MAX CROP",fill="white")
                sheet.paste(tile,((position%columns)*cell_w,(position//columns)*cell_h))
                evidence_rows.append({"community_id":community.community_id,"sheet_rank":position+1,"dimension_id":int(row.dimension_id),"dimension":f"D{int(row.dimension_id):03d}","city_slug":row.city_slug,"pano_index":int(row.pano_index),"patch_row":int(row.patch_row),"patch_column":int(row.patch_column),"activation":float(row.max_patch_activation)})
            sheet.save(figures/"community_evidence"/f"{community.community_id}.jpg",quality=90)
            print(f"evidence [{number:02d}/{len(community_metrics)}] {community.community_id}: {len(selected)} crops",flush=True)
    finally: reader.close()
    pd.DataFrame(evidence_rows).to_csv(data/"semantic/community_evidence_manifest.csv",index=False)
    return {"communities":len(community_metrics),"crop_manifest_rows":len(crop_frame),"evidence_rows":len(evidence_rows)}


def embed(config: dict,force: bool=False) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    import torch
    from torchvision.transforms import v2
    from scripts.multicity.config import MODEL_DIR
    from scripts.multicity.dinov3_vit_backport import load_dinov3_vit
    from scripts.multicity.extract_dinov3_features import MEAN,STD
    cache=config["paths"]["cache_root"]; panorama=config["paths"]["panorama_root"]; target=cache/"validation/dimension_crop_dino_embeddings.f16.npy"
    frame=pd.read_csv(cache/"validation/dimension_crop_manifest.csv")
    if target.exists() and not force:
        existing=np.load(target,mmap_mode="r"); return {"status":"existing","shape":list(existing.shape)}
    model=load_dinov3_vit(MODEL_DIR).eval().requires_grad_(False).to("cuda"); torch.backends.cuda.matmul.allow_tf32=True
    transform=v2.Compose([v2.Resize((224,224),interpolation=v2.InterpolationMode.BICUBIC,antialias=True),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize(mean=MEAN,std=STD)])
    output=np.lib.format.open_memmap(target,mode="w+",dtype=np.float16,shape=(len(frame),768)); manifests={}; reader=TarReader(); batch_size=int(config["visual_validation"]["dino_batch_size"])
    try:
        for start in range(0,len(frame),batch_size):
            tensors=[]
            for row in frame.iloc[start:start+batch_size].itertuples(index=False):
                if row.city_slug not in manifests: manifests[row.city_slug]=load_city_manifest(panorama,row.city_slug)
                group=manifests[row.city_slug].loc[manifests[row.city_slug].pano_index.eq(int(row.pano_index))].sort_values("direction_index"); source=group.iloc[int(row.direction_index)]
                image=reader.image(str(source.tar_path),int(source.jpg_offset),int(source.jpg_size)); crop=crop_from_direction(image,int(row.patch_row),int(row.local_patch_column)); tensors.append(transform(crop))
            pixels=torch.stack(tensors).to("cuda",non_blocking=True)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16): values=model(pixel_values=pixels).last_hidden_state[:,0].float()
            values=torch.nn.functional.normalize(values,dim=1).cpu().numpy(); output[start:start+len(values)]=values.astype(np.float16)
            if start%(batch_size*10)==0 or start+len(values)==len(frame): output.flush(); print(f"DINO crop embedding {start+len(values):,}/{len(frame):,}",flush=True)
    finally: reader.close()
    return {"shape":list(output.shape),"dtype":"float16"}


def analyze(config: dict,force: bool=False) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache=config["paths"]["cache_root"]; data=config["paths"]["paper_data_root"]; figures=config["paths"]["paper_figure_root"]
    frame=pd.read_csv(cache/"validation/dimension_crop_manifest.csv"); embeddings=np.asarray(np.load(cache/"validation/dimension_crop_dino_embeddings.f16.npy",mmap_mode="r"),dtype=np.float32); embeddings=l2_rows(embeddings)
    prototypes=np.zeros((512,768),dtype=np.float64)
    for dimension,indices in frame.groupby("dimension_id").indices.items(): prototypes[int(dimension)]=l2_rows(embeddings[np.asarray(indices)].mean(axis=0,keepdims=True))[0]
    np.save(cache/"validation/dimension_visual_prototypes.f32.npy",prototypes.astype(np.float32)); similarity=prototypes@prototypes.T
    membership=pd.read_csv(data/"leiden/latent_to_fine.csv"); labels=pd.factorize(membership.fine_community,sort=True)[0]; upper=np.triu_indices(512,1); same=labels[upper[0]]==labels[upper[1]]; values=similarity[upper]
    within=values[same]; between=values[~same]; difference=float(within.mean()-between.mean()); pooled=math.sqrt(((len(within)-1)*within.var(ddof=1)+(len(between)-1)*between.var(ddof=1))/(len(within)+len(between)-2)); effect=difference/max(pooled,1e-12)
    rng=np.random.default_rng(int(config["random_seed"])); permutations=[]; base=labels.copy(); n_perm=int(config["visual_validation"]["n_permutations"])
    for index in range(n_perm):
        shuffled=rng.permutation(base); mask=shuffled[upper[0]]==shuffled[upper[1]]; permutations.append(float(values[mask].mean()))
    permutations=np.asarray(permutations); p_value=float((1+(permutations>=within.mean()).sum())/(n_perm+1))
    community_rows=[]
    for community,group in membership.groupby("fine_community"):
        ids=group.dimension_id.to_numpy(int); local=similarity[np.ix_(ids,ids)]; local_values=local[np.triu_indices(len(ids),1)] if len(ids)>1 else np.asarray([]); other=np.flatnonzero(~np.isin(np.arange(512),ids)); external=similarity[np.ix_(ids,other)]
        community_rows.append({"community_id":community,"dimensions":len(ids),"mean_within_similarity":float(local_values.mean()) if len(local_values) else np.nan,"mean_external_similarity":float(external.mean()),"difference":float(local_values.mean()-external.mean()) if len(local_values) else np.nan})
    pd.DataFrame(community_rows).to_csv(data/"validation/visual_coherence.csv",index=False); pd.DataFrame({"permutation":range(n_perm),"random_within_similarity":permutations}).to_csv(data/"validation/permutation_results.csv",index=False)
    summary={"embedding":"frozen DINOv3 CLS embedding of each dimension's ten top-activated panorama crops; embeddings excluded from discovery","mean_within_community_similarity":float(within.mean()),"mean_between_community_similarity":float(between.mean()),"difference":difference,"cohens_d":effect,"permutation_p_value":p_value,"n_permutations":n_perm}
    save_json(data/"validation/visual_coherence_summary.json",summary)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True); ax.hist(permutations,bins=35,color="#999999",alpha=.8,label="Random size-preserving partitions"); ax.axvline(within.mean(),color="#D55E00",lw=2,label=f"Observed = {within.mean():.3f}"); ax.set(xlabel="Mean within-community DINO similarity",ylabel="Permutations",title="Independent image-space coherence"); ax.legend(frameon=False); fig.savefig(figures/"visual_coherence.png",dpi=240); plt.close(fig)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--phase",choices=["all","prepare","embed","analyze"],default="all"); parser.add_argument("--force",action="store_true"); args=parser.parse_args(); config=load_config(args.config)
    if args.phase=="all":
        base=[sys.executable,"-m","scripts.multicity.graph_visual_vocabulary.06_validate_visual_coherence","--config",str(args.config)]
        for phase in ("prepare","embed","analyze"): subprocess.run([*base,"--phase",phase,*(["--force"] if args.force else [])],check=True)
    else: print({"prepare":prepare,"embed":embed,"analyze":analyze}[args.phase](config,args.force))


if __name__=="__main__": main()
