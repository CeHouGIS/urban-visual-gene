"""Small CNN image-QC classifier {good, dark, blur}, trained on SYNTHETIC
corruptions of clean street-view images (no manual labels). More robust than
fixed Laplacian thresholds for blur/dark. Tunnel stays on CLIP (semantic).

  python -m formal.qc_net --train                       # gen synthetic + train -> qc_net.pt
  python -m formal.qc_net --apply --glob 'thumbs/*.jpg' # predict + calibration montages
"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
RES=160
G2=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2")
DIAG=G2/"genes"/"diag"; MODEL=DIAG/"qc_net.pt"
def log(*a): import time;print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

# ---------- synthetic corruptions ----------
def motion(im,k,rng):
    a=np.asarray(im).astype(np.float32); ker=np.zeros((k,k),np.float32)
    if rng.random()<0.5: ker[k//2,:]=1.0/k
    else: np.fill_diagonal(ker,1.0/k)
    for c in range(3): a[...,c]=ndimage.convolve(a[...,c],ker,mode="reflect")
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def corrupt(im,lab,rng):
    if lab==0: return im                                  # good
    if lab==1:                                            # dark
        a=np.asarray(im).astype(np.float32)
        if rng.random()<0.35: a*=rng.uniform(0.0,0.06)     # near-black
        else: a*=rng.uniform(0.06,0.35)                    # dim
        return Image.fromarray(a.clip(0,255).astype(np.uint8))
    t=rng.integers(0,3)                                   # blur
    if t==0: b=im.filter(ImageFilter.GaussianBlur(rng.uniform(2.0,6.0)))
    elif t==1: b=motion(im,int(rng.integers(7,19)),rng)
    else:
        s=max(4,int(RES*rng.uniform(0.12,0.4))); b=im.resize((s,s)).resize((RES,RES))
    if rng.random()<0.45:                                 # LARGE-region blur (partial)
        a=np.asarray(im).copy(); bb=np.asarray(b); h=int(rng.integers(int(RES*0.5),RES))
        if rng.random()<0.5: a[:h]=bb[:h]
        else: a[-h:]=bb[-h:]
        return Image.fromarray(a)
    return b

def aug_tunnel(im,rng):                                   # diversify the small real-tunnel set
    if rng.random()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random()<0.7:
        s=rng.uniform(0.6,1.0); w=int(RES*s)
        x=int(rng.integers(0,RES-w+1)); y=int(rng.integers(0,RES-w+1))
        im=im.crop((x,y,x+w,y+w)).resize((RES,RES))
    a=np.asarray(im,np.float32)*rng.uniform(0.6,1.3)      # brightness
    a=(a-128)*rng.uniform(0.8,1.3)+128                    # contrast
    if rng.random()<0.25: a=a+rng.normal(0,8,a.shape)     # noise
    return Image.fromarray(a.clip(0,255).astype(np.uint8))

class QCData(Dataset):
    def __init__(s,paths,tun,hn,n): s.p=paths; s.tun=tun; s.hn=hn; s.n=n
    def __len__(s): return s.n
    def __getitem__(s,i):
        rng=np.random.default_rng()
        lab=int(rng.integers(0,4))
        if lab==3:                                        # tunnel (human-labeled positives)
            p=s.tun[int(rng.integers(len(s.tun)))]
            im=aug_tunnel(Image.open(p).convert("RGB").resize((RES,RES)),rng)
        elif lab==0 and s.hn and rng.random()<0.5:        # HARD NEGATIVES -> good (tunnel-lookalikes, not tunnel)
            im=Image.open(s.hn[int(rng.integers(len(s.hn)))]).convert("RGB").resize((RES,RES))
            if rng.random()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
        else:                                             # good/dark/blur (synthetic on clean)
            im=Image.open(s.p[i%len(s.p)]).convert("RGB").resize((RES,RES))
            if rng.random()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
            im=corrupt(im,lab,rng)
        x=torch.from_numpy(np.asarray(im,np.float32)/255).permute(2,0,1)
        return x,lab

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        blk=lambda i,o:nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(3,16),blk(16,32),blk(32,64),blk(64,96),nn.AdaptiveAvgPool2d(1))
        s.fc=nn.Linear(96,4)
    def forward(s,x): return s.fc(s.f(x).flatten(1))
CLASSES=["good","dark","blur","tunnel"]

def clean_paths():
    import pandas as pd
    df=pd.read_parquet(DIAG/"qc_metrics.parquet")
    ok=df[(df.bright.between(85,190))&(df.lapvar>350)&(df.blur_tilefrac<0.05)&(df.skytop.between(95,235))]
    return ok["path"].tolist()

def tunnel_paths():
    f=DIAG/"qc_tunnel_pos.txt"                            # human-labeled clean tunnels (preferred)
    if f.exists(): return [l.strip() for l in open(f) if l.strip()]
    import pandas as pd; df=pd.read_parquet(DIAG/"qc_clip_scores.parquet")
    return df[df.tunnel_prob>0.55]["path"].tolist()
def hardneg_paths():
    f=DIAG/"qc_tunnel_neg.txt"                            # human-labeled tunnel-lookalike NON-tunnels
    return [l.strip() for l in open(f) if l.strip()] if f.exists() else []

def train():
    paths=clean_paths(); tun=tunnel_paths(); hn=hardneg_paths()
    log(f"{len(paths)} clean sources, {len(tun)} labeled tunnels, {len(hn)} hard negatives")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),5e-4,weight_decay=1e-4)
    EP=16; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP)
    tr=DataLoader(QCData(paths,tun,hn,9000),batch_size=64,shuffle=True,num_workers=8,drop_last=True)
    va=DataLoader(QCData(paths,tun,hn,2400),batch_size=64,num_workers=4)
    best=-1
    for ep in range(EP):
        net.train(); tl=0
        for x,y in tr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad()
            loss=nn.functional.cross_entropy(net(x),y); loss.backward(); opt.step(); tl+=loss.item()
        sch.step()
        net.eval(); cor=np.zeros(4); tot=np.zeros(4)
        with torch.no_grad():
            for x,y in va:
                p=net(x.to(dev)).argmax(1).cpu().numpy(); y=y.numpy()
                for c in range(4): tot[c]+=(y==c).sum(); cor[c]+=((p==c)&(y==c)).sum()
        acc=cor/np.maximum(tot,1); score=acc.min()                 # pick epoch with best WORST-class acc
        star=""
        if score>best: best=score; torch.save(net.state_dict(),MODEL); star=" *saved"
        log(f"ep{ep} loss {tl/len(tr):.3f}  val "+" ".join(f"{CLASSES[c]} {acc[c]:.2f}" for c in range(4))+f"  min {score:.2f}{star}")
    log(f"best min-class acc {best:.2f} -> {MODEL}")

@torch.no_grad()
def predict(paths,batch=128):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=Net().to(dev); net.load_state_dict(torch.load(MODEL,map_location=dev)); net.eval()
    out=np.zeros((len(paths),4),np.float32)
    for i in range(0,len(paths),batch):
        ims=[]
        for p in paths[i:i+batch]:
            try: ims.append(np.asarray(Image.open(p).convert("RGB").resize((RES,RES)),np.float32)/255)
            except: ims.append(np.zeros((RES,RES,3),np.float32))
        x=torch.from_numpy(np.stack(ims)).permute(0,3,1,2).to(dev)
        out[i:i+len(ims)]=net(x).softmax(1).cpu().numpy()
        if i%2560==0 and i: log(f"  {i}/{len(paths)}")
    return out

def montage(paths,idx,title,out):
    from PIL import ImageDraw
    idx=idx[:48]; cols=8; rows=(len(idx)+cols-1)//cols; TH=140
    cv=Image.new("RGB",(cols*TH,rows*TH+22),(12,12,18)); dr=ImageDraw.Draw(cv); dr.text((6,4),title,fill=(230,230,240))
    for i,k in enumerate(idx):
        try: im=Image.open(paths[k]).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(30,0,0))
        cv.paste(im,((i%cols)*TH,22+(i//cols)*TH))
    cv.save(out); log("montage ->"+str(out))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",action="store_true")
    ap.add_argument("--apply",action="store_true"); ap.add_argument("--glob"); ap.add_argument("--thr",type=float,default=0.5)
    a=ap.parse_args()
    if a.train: train()
    if a.apply:
        import glob as _glob; paths=sorted(_glob.glob(a.glob,recursive=True))
        P=predict(paths); pred=P.argmax(1)
        log("  ".join(f"{CLASSES[c]} {int((pred==c).sum())}" for c in range(4)))
        montage(paths,np.argsort(-P[:,1]),"predicted DARK (top P)",DIAG/"qcnet_dark.png")
        montage(paths,np.argsort(-P[:,2]),"predicted BLUR (top P)",DIAG/"qcnet_blur.png")
        montage(paths,np.argsort(-P[:,3]),"predicted TUNNEL (top P)",DIAG/"qcnet_tunnel.png")
        import pandas as pd
        pd.DataFrame({"path":paths,"p_good":P[:,0],"p_dark":P[:,1],"p_blur":P[:,2],"p_tunnel":P[:,3]}).to_parquet(DIAG/"qcnet_pred.parquet")
if __name__=="__main__": main()
