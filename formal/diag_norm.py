"""GPU diagnostic: are DINOv3 @448 border patches high-norm artifact tokens?
Maps per-patch RAW (pre-L2-normalize) token norm over the 28x28 grid."""
import os, numpy as np, torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
MID="facebook/dinov3-vitl16-pretrain-lvd1689m"
TOK=(Path.home()/".cache/huggingface/token").read_text().strip()
IMG=Path("/global/scratch/users/cehou/data/SVIs/GSV/images/China/HongKong")
RES=448; G=28
dev="cuda" if torch.cuda.is_available() else "cpu"
proc=AutoImageProcessor.from_pretrained(MID,token=TOK)
model=AutoModel.from_pretrained(MID,token=TOK).eval().to(dev)
if dev=="cuda": model=model.to(torch.bfloat16)
with torch.no_grad():
    d=proc(images=Image.new("RGB",(RES,RES)),return_tensors="pt",size={"height":RES,"width":RES}).to(dev)
    if dev=="cuda": d["pixel_values"]=d["pixel_values"].to(torch.bfloat16)
    T=model(**d).last_hidden_state.shape[1]
prefix=T-G*G
print(f"prefix={prefix} tokens={T} dev={dev}",flush=True)

paths=[]
for root,dirs,files in os.walk(IMG):
    dirs.sort()
    for f in sorted(files):
        if f.endswith("_0.jpg"): paths.append(os.path.join(root,f))
        if len(paths)>=16: break
    if len(paths)>=16: break

acc=np.zeros((G,G))
for p in paths:
    im=Image.open(p).convert("RGB").resize((RES,RES),Image.BILINEAR)
    x=proc(images=im,return_tensors="pt",size={"height":RES,"width":RES}).to(dev)
    if dev=="cuda": x["pixel_values"]=x["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        h=model(**x).last_hidden_state[0,prefix:].float().cpu()
    acc+=h.norm(dim=1).reshape(G,G).numpy()
acc/=len(paths); rel=acc/acc.mean()
print(f"\nmean per-patch norm map (rel to mean), {len(paths)} imgs:",flush=True)
for r in range(G): print(" ".join(f"{rel[r,c]:3.1f}" for c in range(G)),flush=True)
print(f"\ncol rel-norm: c0={rel[:,0].mean():.1f} c1={rel[:,1].mean():.1f} c2={rel[:,2].mean():.1f} "
      f"interior={rel[:,5:23].mean():.1f} c25={rel[:,25].mean():.1f} c26={rel[:,26].mean():.1f} c27={rel[:,27].mean():.1f}",flush=True)
print(f"row rel-norm: r0={rel[0].mean():.1f} r1={rel[1].mean():.1f} interior={rel[5:23].mean():.1f} r26={rel[26].mean():.1f} r27={rel[27].mean():.1f}",flush=True)
hi=acc>1.5*acc.mean()
bc=hi[:,[0,1,2,25,26,27]].sum()+hi[[0,1,26,27],:].sum()
print(f"\nhigh-norm(>1.5x mean): {int(hi.sum())}/{G*G} patches; {bc} on outer ring "
      f"({bc/max(hi.sum(),1):.0%} of high-norm are border)",flush=True)
print("VERDICT:", "border artifacts ARE high-norm -> norm filter viable"
      if rel[:,0].mean()>1.4 or rel[:,27].mean()>1.4 else
      "border artifacts NOT clearly high-norm -> need position/direction fix",flush=True)
