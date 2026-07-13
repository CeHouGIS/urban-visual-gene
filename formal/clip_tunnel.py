"""CLIP zero-shot TUNNEL detector: P(tunnel) from CLIP image-text similarity vs
tunnel/street prompts — semantic, far more accurate than the brightness heuristic.
GPU if available (CPU fallback). Downloads openai/clip-vit-base-patch32 (cache once).

  # calibrate: montage sorted by CLIP tunnel prob + distribution
  python -m formal.clip_tunnel --glob 'thumbs/*.jpg' --calibrate
  # update per-city QC parquets: tunnel = CLIP prob > THRESH ; is_bad = black|blur|tunnel
  python -m formal.clip_tunnel --update-qc --thresh 0.55
"""
import os, sys, argparse, numpy as np, torch
from pathlib import Path
from PIL import Image
MID="openai/clip-vit-base-patch32"
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
TUNNEL=["a photo taken inside a road tunnel","the interior of a highway tunnel",
        "inside a dark tunnel with overhead lights","an underground road tunnel","driving through a tunnel"]
STREET=["a street view of a city road","an outdoor street with buildings and shops",
        "a road with sky and trees","a suburban residential street","a highway with open sky",
        "a park with vegetation","a night street scene"]
def log(*a): import time;print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

def load():
    from transformers import CLIPModel, CLIPProcessor
    tok=(Path.home()/".cache/huggingface/token").read_text().strip()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=CLIPModel.from_pretrained(MID,token=tok).to(dev).eval()
    proc=CLIPProcessor.from_pretrained(MID,token=tok,use_fast=True)
    with torch.no_grad():
        t=proc(text=TUNNEL+STREET,return_tensors="pt",padding=True).to(dev)
        to=m.text_model(input_ids=t["input_ids"],attention_mask=t.get("attention_mask"))
        te=m.text_projection(to.pooler_output); te=te/te.norm(dim=-1,keepdim=True)
    log(f"CLIP loaded dev={dev}, {len(TUNNEL)} tunnel + {len(STREET)} street prompts")
    return m,proc,te,dev

def score(paths,m,proc,te,dev,batch=256):
    ntun=len(TUNNEL); out=np.zeros(len(paths),np.float32)
    for i in range(0,len(paths),batch):
        ims=[]
        for p in paths[i:i+batch]:
            try: ims.append(Image.open(p).convert("RGB"))
            except: ims.append(Image.new("RGB",(224,224)))
        px=proc(images=ims,return_tensors="pt").to(dev)
        with torch.no_grad():
            vo=m.vision_model(pixel_values=px["pixel_values"])
            ie=m.visual_projection(vo.pooler_output); ie=ie/ie.norm(dim=-1,keepdim=True)
            sm=(m.logit_scale.exp()*ie@te.T).softmax(1)
            out[i:i+len(ims)]=sm[:,:ntun].sum(1).float().cpu().numpy()
        if i%2560==0 and i: log(f"  scored {i:,}/{len(paths):,}")
    return out

def montage(paths,scores,title,out,n=48):
    o=np.argsort(-scores)[:n]; cols=8; rows=(len(o)+cols-1)//cols; TH=140
    from PIL import ImageDraw
    cv=Image.new("RGB",(cols*TH,rows*TH+22),(12,12,18)); dr=ImageDraw.Draw(cv); dr.text((6,4),title,fill=(230,230,240))
    for i,k in enumerate(o):
        try: im=Image.open(paths[k]).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(30,0,0))
        cv.paste(im,((i%cols)*TH,22+(i//cols)*TH)); dr.text(((i%cols)*TH+3,22+(i//cols)*TH+3),f"{scores[k]:.2f}",fill=(255,255,120))
    cv.save(out); log("montage ->"+str(out))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob"); ap.add_argument("--calibrate",action="store_true")
    ap.add_argument("--update-qc",action="store_true"); ap.add_argument("--thresh",type=float,default=0.55)
    ap.add_argument("--diag",default=str(REPO/"formal"/"formal_out_global2"/"genes"/"diag"))
    a=ap.parse_args()
    m,proc,te,dev=load()
    import pandas as pd, glob as _glob
    if a.glob:
        paths=sorted(_glob.glob(a.glob,recursive=True)); log(f"scoring {len(paths):,} imgs")
        sc=score(paths,m,proc,te,dev)
        print(f"tunnel_prob p50/75/90/95/99: "+" ".join(f"{v:.2f}" for v in np.percentile(sc,[50,75,90,95,99])))
        print(f"  >0.5: {(sc>0.5).sum()}  >0.6: {(sc>0.6).sum()}  >0.7: {(sc>0.7).sum()}")
        if a.calibrate: montage(paths,sc,"most tunnel-like by CLIP",Path(a.diag)/"qc_clip_tunnel.png")
    if a.update_qc:
        from formal.gpu_run import imgpath, CITY_DIR
        QC=REPO/"formal"/"qc"; tot=0
        for pf in sorted(QC.glob("*.parquet")):
            city=pf.stem
            if city not in CITY_DIR: continue
            df=pd.read_parquet(pf); paths=[str(imgpath(city,p,0)) for p in df["pano_id"]]
            sc=score(paths,m,proc,te,dev)
            df["clip_tunnel"]=sc; df["tunnel"]=sc>a.thresh
            df["is_bad"]=df["black"]|df["blur"]|df["tunnel"]
            df.to_parquet(pf); tot+=int(df["tunnel"].sum())
            log(f"{city:10s} CLIP tunnel {int((sc>a.thresh).sum()):3d} (was heuristic) -> bad {int(df['is_bad'].sum()):3d}")
        log(f"[done] CLIP tunnel updated all cities, total {tot:,} tunnels (thresh {a.thresh})")

if __name__=="__main__": main()
