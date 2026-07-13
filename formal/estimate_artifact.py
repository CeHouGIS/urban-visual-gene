"""Estimate the TRUE border-artifact directions as mean border-column features
(content averages out, positional artifact remains), verify projection removes
the border consistency, and save dirs for the retrain. GPU."""
import os, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
MID="facebook/dinov3-vitl16-pretrain-lvd1689m"
TOK=(Path.home()/".cache/huggingface/token").read_text().strip()
IMG=Path("/global/scratch/users/cehou/data/SVIs/GSV/images/China/HongKong")
OUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out")
RES=448; G=28
dev="cuda" if torch.cuda.is_available() else "cpu"
proc=AutoImageProcessor.from_pretrained(MID,token=TOK)
model=AutoModel.from_pretrained(MID,token=TOK).eval().to(dev)
if dev=="cuda": model=model.to(torch.bfloat16)
with torch.no_grad():
    d=proc(images=Image.new("RGB",(RES,RES)),return_tensors="pt",size={"height":RES,"width":RES}).to(dev)
    if dev=="cuda": d["pixel_values"]=d["pixel_values"].to(torch.bfloat16)
    T=model(**d).last_hidden_state.shape[1]
prefix=T-G*G; print(f"prefix={prefix} dev={dev}",flush=True)

paths=[]
for root,dirs,files in os.walk(IMG):
    dirs.sort()
    for f in sorted(files):
        if f.endswith("_0.jpg"): paths.append(os.path.join(root,f))
        if len(paths)>=64: break
    if len(paths)>=64: break

H=[]
for p in paths:
    im=Image.open(p).convert("RGB").resize((RES,RES),Image.BILINEAR)
    x=proc(images=im,return_tensors="pt",size={"height":RES,"width":RES}).to(dev)
    if dev=="cuda": x["pixel_values"]=x["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        h=model(**x).last_hidden_state[0,prefix:].float().cpu().reshape(G,G,-1)
    H.append(h)
H=torch.stack(H)                                       # (N, G, G, D)
N,_,_,D=H.shape; print(f"{N} imgs, D={D}",flush=True)

# TRUE artifact dirs = mean feature of extreme border columns (content averages out)
art_left =H[:,:,0,:].reshape(-1,D).mean(0)             # leftmost col mean
art_right=H[:,:,G-1,:].reshape(-1,D).mean(0)           # rightmost col mean
A=F.normalize(torch.stack([art_left,art_right]),dim=1) # (2, D)
np.save(OUT/"artifact_dirs_mean.npy", A.numpy().astype(np.float32))

# compare to SAE atoms #57/#306
sae=torch.load(OUT/"sae_448_k512.pt",map_location="cpu")
atoms=sae["state"]["dec.weight"].T                     # (512, D)
print(f"cos(mean-left , atom#57 )={F.cosine_similarity(A[0],atoms[57],0):.2f}",flush=True)
print(f"cos(mean-right, atom#306)={F.cosine_similarity(A[1],atoms[306],0):.2f}",flush=True)

# verify: projection removes border consistency
Q,_=torch.linalg.qr(A.T); P=Q.T                        # orthonormal (2,D)
def consistency(col):                                  # how aligned are patches in a column across images
    v=F.normalize(H[:,:,col,:].reshape(-1,D),dim=1); m=F.normalize(v.mean(0),dim=0)
    return (v@m).mean().item()                         # 1=identical dir, ~0=content-varied
def proj(v): return v - (v@P.t())@P
def consistency_proj(col):
    v=F.normalize(proj(H[:,:,col,:].reshape(-1,D)),dim=1); m=F.normalize(v.mean(0),dim=0)
    return (v@m).mean().item()
print("\ncolumn direction-consistency (1=artifact-locked, low=real content):",flush=True)
for c in [0,1,2,13,25,26,27]:
    print(f"  col {c:2d}: before={consistency(c):.2f}  after-projection={consistency_proj(c):.2f}",flush=True)
print("saved artifact_dirs_mean.npy",flush=True)
