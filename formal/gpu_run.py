"""Formal patch-level SAE run — GPU, 448 resolution, 4-city shared dictionary.

Stages (each skipped if its output exists -> resumable on requeue):
  S0/S1  build a balanced patch sample from 4 cities @448, train a shared
         Top-K SAE (K, topk configurable).
  S2     for street-analysis panos (roads with on-disk 4-heading coverage),
         extract @448, encode with the SAE, save argmax gene maps + thumbnails
         + per-point metadata per city.

Downstream morphotype clustering + web assets are regenerated on the login node
from S2 outputs (CPU-only, no GPU needed).

  python -m formal.gpu_run --dict-panos 4000 --K 2048 --topk 32 --epochs 60 \
      --street-roads 60 --street-pts 5
"""
import os, sys, json, time, random, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MID   = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TOK   = (Path.home()/".cache/huggingface/token").read_text().strip()
REPO  = Path("/global/scratch/users/cehou/urban-visual-gene")
IMROOT= Path("/global/scratch/users/cehou/data/SVIs/GSV/images")
OUT   = Path(os.environ.get("FORMAL_OUT", REPO/"formal"/"formal_out")); OUT.mkdir(parents=True, exist_ok=True)
CITY_DIR = {"HongKong":"China/HongKong","Singapore":"Singapore/Singapore",
            "Amsterdam":"Netherlands/Amsterdam","CapeTown":"SouthAfrica/CapeTown",
            # 9 additional download-complete cities (dictionary only; no road data yet)
            "Paris":"France/Paris","SaoPaulo":"Brazil/SaoPaulo","MexicoCity":"Mexico/MexicoCity",
            "Sydney":"Australia/Sydney","Jakarta":"Indonesia/Jakarta","Dhaka":"Bangladesh/Dhaka",
            "NewDelhi":"India/NewDelhi","Manila":"Philippines/Manila","Vienna":"Austria/Vienna"}
HEADINGS=[0,90,180,270]

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
def imgpath(city,pid,h):
    # download pipeline LOWERCASES the 3-char dir prefix (letters); filenames keep
    # original case. Using raw pid[0:3] silently missed ~79% of panos (any uppercase
    # letter in first 3 chars). Lowercase the prefix to match on-disk dirs.
    return IMROOT/CITY_DIR[city]/pid[0].lower()/pid[1].lower()/pid[2].lower()/f"{pid}_{h}.jpg"

# ─────────────────────────── DINOv3 batched extractor ───────────────────────
class Extractor:
    def __init__(self, res, art_factor=0.0, proj_path=None):
        from transformers import AutoImageProcessor, AutoModel
        self.res=res; self.G=res//16; self.art_factor=art_factor
        self.P=None
        if proj_path and Path(proj_path).exists():
            A=torch.tensor(np.load(proj_path),dtype=torch.float32)          # (m, D)
            Q,_=torch.linalg.qr(A.T); self.P=Q.T                            # orthonormal rows (m, D)
            log(f"projecting out {self.P.shape[0]} artifact directions from features")
        self.dev="cuda" if torch.cuda.is_available() else "cpu"
        self.proc=AutoImageProcessor.from_pretrained(MID, token=TOK)
        self.model=AutoModel.from_pretrained(MID, token=TOK).to(self.dev).eval()
        if self.dev=="cuda": self.model=self.model.to(torch.bfloat16)
        from PIL import Image
        with torch.no_grad():
            d=self.proc(images=Image.new("RGB",(res,res)),return_tensors="pt",
                        size={"height":res,"width":res})
            px=d["pixel_values"].to(self.dev)
            if self.dev=="cuda": px=px.to(torch.bfloat16)
            T=self.model(px).last_hidden_state.shape[1]
        self.prefix=T-self.G*self.G
        log(f"extractor res={res} grid={self.G}x{self.G} tokens={T} prefix={self.prefix} dev={self.dev}")

    @torch.no_grad()
    def batch(self, pils):
        """list[PIL] -> (feats (B,G*G,1024) L2-normed, artifact_mask (B,G*G) bool), cpu.
        Artifact = RAW (pre-normalize) token norm > art_factor * per-image mean norm
        (ViT register-artifact tokens are high-norm; detected here BEFORE normalization,
        which otherwise erases the norm signal). art_factor<=0 disables (mask all False)."""
        d=self.proc(images=pils,return_tensors="pt",size={"height":self.res,"width":self.res})
        px=d["pixel_values"].to(self.dev)
        if self.dev=="cuda": px=px.to(torch.bfloat16)
        h=self.model(px).last_hidden_state[:,self.prefix:,:].float()      # (B, G*G, 1024) RAW
        if self.P is not None:                                            # ROOT FIX: remove
            Pd=self.P.to(h.device)                                        # positional-artifact subspace
            h=h - (h @ Pd.t()) @ Pd                                       # z' = z - (z·A)A
        nrm=h.norm(dim=2)
        if self.art_factor>0:
            art=nrm > self.art_factor*nrm.mean(dim=1,keepdim=True)
        else:
            art=torch.zeros_like(nrm,dtype=torch.bool)
        return F.normalize(h,dim=2).cpu(), art.cpu()

def load_pils(items, res):
    """items: list of (city,pid,h). Returns (pils, kept_idx) skipping unreadable."""
    from PIL import Image
    pils, kept=[],[]
    for j,(city,pid,h) in enumerate(items):
        try:
            pils.append(Image.open(imgpath(city,pid,h)).convert("RGB").resize((res,res),Image.BILINEAR))
            kept.append(j)
        except Exception: pass
    return pils, kept

# ─────────────────────────── Top-K SAE ──────────────────────────────────────
class SAE(nn.Module):
    def __init__(s,D,K,topk):
        super().__init__(); s.D,s.K,s.topk=D,K,topk
        s.b_pre=nn.Parameter(torch.zeros(D)); s.enc=nn.Linear(D,K); s.dec=nn.Linear(K,D,bias=False)
        with torch.no_grad():
            nn.init.normal_(s.dec.weight); s._norm()
            s.enc.weight.copy_(s.dec.weight.T.clone()); s.enc.bias.zero_()
    def _norm(s):
        with torch.no_grad():
            w=s.dec.weight; s.dec.weight.copy_(w/w.norm(dim=0,keepdim=True).clamp(min=1e-8))
    def encode(s,z):
        pre=s.enc(z-s.b_pre); val,idx=pre.topk(s.topk,dim=1)
        a=torch.zeros_like(pre); a.scatter_(1,idx,F.relu(val)); return a
    def forward(s,z):
        a=s.encode(z); return a, s.dec(a)+s.b_pre

# ─────────────────────────── data helpers ───────────────────────────────────
import pandas as pd
def city_panos(city):
    d=REPO/"outputs"/CITY_DIR[city]
    rmp=pd.read_parquet(d/"road_matched_panos.parquet")
    keep=[c for c in ["pano_id","matched_road_id","chainage_m","lat","lon"] if c in rmp.columns]
    rmp=rmp[keep]
    if "lat" not in rmp.columns:
        pf=pd.read_parquet(d/"pano_features.parquet",columns=["pano_id","lat","lon"])
        rmp=rmp.merge(pf,on="pano_id",how="left")
    return rmp

def disk_panos(city, n, rng, stride=4):
    """Collect pano_ids straight from the image dir (fallback when no meta DB).
    Scans heading-0 jpgs, samples n."""
    root=IMROOT/CITY_DIR[city]; ids=[]
    for r,dirs,files in os.walk(root):
        dirs.sort()
        for f in files:
            if f.endswith("_0.jpg"): ids.append(f[:-6])
        if len(ids)>=n*stride: break
    rng.shuffle(ids); return ids[:n]

def _db(city, sub, fname):
    country=CITY_DIR[city].split("/")[0]
    return f"/global/scratch/users/cehou/data/SVIs/GSV/metadata/{country}/{city}/{sub}/{fname.format(country=country,city=city)}"
def meta_db(city): return _db(city,"meta","{country}_{city}.db")
def points_db(city): return _db(city,"sampling_points_db","combined.db")

def bad_panos(city):
    """QC blocklist: pano_ids flagged black/blur/tunnel by quality_filter -> qc/<city>.parquet.
    Empty set if no QC has been run for the city (so sampling is unaffected until then)."""
    if not hasattr(bad_panos,"_c"): bad_panos._c={}
    if city not in bad_panos._c:
        f=REPO/"formal"/"qc"/f"{city}.parquet"; s=set()
        if f.exists():
            try:
                import pandas as pd; d=pd.read_parquet(f); s=set(d.loc[d["is_bad"],"pano_id"].astype(str))
                log(f"  {city}: QC blocklist {len(s):,} panos")
            except Exception as e: log(f"  {city}: QC load failed {e}")
        bad_panos._c[city]=s
    return bad_panos._c[city]

def stratified_panos(city, quota, seed=0, grid=3):
    """DE-BIASED spatial sampling. Weight each pano by (road presence in its ~100m
    cell) / (panos in its cell), so GSV over-capture on busy roads is corrected and
    the sample tracks the STREET NETWORK, not capture density. Road presence per cell
    = count of sampling points (placed evenly along roads ∝ road length). Weighted
    sample without replacement. Falls back to disk walk if the meta DB is empty (Vienna)."""
    import sqlite3, collections
    try:
        con=sqlite3.connect(f"file:{meta_db(city)}?mode=ro&immutable=1",uri=True)
        panos=con.execute("SELECT panoid,lat,lon FROM gsv WHERE download=1").fetchall(); con.close()
    except Exception: panos=[]
    panos=[p for p in panos if p[1] is not None and p[2] is not None]
    if len(panos) < quota:
        log(f"  {city}: meta DB {len(panos)} panos (<quota) — disk-walk fallback")
        dp=disk_panos(city, quota, random.Random(seed)); bad=bad_panos(city)
        return [p for p in dp if str(p) not in bad] if bad else dp
    cell=lambda la,lo:(round(la,grid),round(lo,grid))
    # road presence per cell from sampling points (∝ road length); glob all *.db
    # (naming varies: combined.db vs {city}_{country}_roads_N.db)
    import glob
    country=CITY_DIR[city].split("/")[0]
    pdir=f"/global/scratch/users/cehou/data/SVIs/GSV/metadata/{country}/{city}/sampling_points_db"
    roadw=collections.Counter(); got=False
    for db in glob.glob(f"{pdir}/*.db"):
        try:
            con=sqlite3.connect(f"file:{db}?mode=ro&immutable=1",uri=True)
            for la,lo in con.execute("SELECT lat,lon FROM points"):
                if la is not None and lo is not None: roadw[cell(la,lo)]+=1
            con.close(); got=True
        except Exception: pass
    pcell=[cell(la,lo) for _,la,lo in panos]
    ppc=collections.Counter(pcell)
    if got:
        w=np.array([(roadw.get(c,0)+1.0)/ppc[c] for c in pcell],float)   # road-weighted
        mode="road-weighted"
    else:
        w=np.array([1.0/ppc[c] for c in pcell],float)                    # uniform-per-cell
        mode="uniform-cell"
    w=w/w.sum()
    ncand=min(len(panos), 6*quota)                                       # oversample; on-disk yield ~21%
    idx=np.random.default_rng(seed).choice(len(panos),size=ncand,replace=False,p=w)
    log(f"  {city}: {len(panos):,} panos, {len(ppc):,} cells, {mode} -> {ncand:,} candidates")
    cand=[panos[i][0] for i in idx]; bad=bad_panos(city)
    if bad:
        n0=len(cand); cand=[p for p in cand if str(p) not in bad]
        log(f"  {city}: QC dropped {n0-len(cand):,} low-quality panos -> {len(cand):,}")
    return cand

# ─────────────────────────── S0: dictionary patch sample (once) ─────────────
def build_sample(ext, args):
    sp=OUT/"dict_sample.f16.npy"
    if sp.exists():
        Z=np.load(sp); log(f"S0 reuse sample {Z.shape}"); return Z
    rng=random.Random(0)
    smap=getattr(args,"sample_map",None) or {}
    log(f"S0 collecting dictionary patch sample (heading 0), per-city={'sample_counts.json' if smap else args.dict_panos} ...")
    for city in args.cities:
        cf=OUT/f"dict_{city}.f16.npy"
        if cf.exists():                                      # per-city checkpoint (resume on requeue)
            log(f"  {city}: cached {np.load(cf,mmap_mode='r').shape[0]:,} patches — skip"); continue
        per=smap.get(city, args.dict_panos)
        cands=stratified_panos(city, per, seed=0)            # ~6x oversample (weighted)
        ondisk=[]                                            # Phase 1: verify on disk, keep `per`
        for pid in cands:
            if imgpath(city,pid,0).exists(): ondisk.append(pid)
            if len(ondisk)>=per: break
        log(f"  {city}: {len(ondisk):,}/{per:,} on-disk panos from {len(cands):,} candidates")
        items=[(city,pid,h) for pid in ondisk for h in HEADINGS]   # Phase 2: ALL 4 headings
        buf=[]
        for i in range(0, len(items), args.batch):
            pils,kept=load_pils(items[i:i+args.batch], args.res)
            if not pils: continue
            pt,art=ext.batch(pils); pt=pt.numpy(); art=art.numpy()
            for r in range(pt.shape[0]):
                ok=np.where(~art[r])[0]                       # drop artifact patches from training
                if len(ok)==0: continue
                sel=rng.sample(list(ok),min(args.keep_patches,len(ok)))
                buf.append(pt[r,sel].astype(np.float16))
            if (i//args.batch)%40==0: log(f"  {city}: {i:,}/{len(items):,} imgs")
        if buf:
            cz=np.concatenate(buf,0); np.save(cf,cz); log(f"  {city}: {cz.shape[0]:,} patches -> {cf.name}")
    parts=[np.load(OUT/f"dict_{c}.f16.npy") for c in args.cities if (OUT/f"dict_{c}.f16.npy").exists()]
    Z=np.concatenate(parts,0); np.save(sp,Z)
    log(f"S0 done: {Z.shape[0]:,} patches x {Z.shape[1]} -> {sp.name}")
    return Z

# ─────────────────────────── S1: train one SAE per K ────────────────────────
def train_sae(Z, K, args):
    saep=OUT/f"sae_{args.res}_k{K}.pt"
    if saep.exists(): log(f"S1 skip K={K} — exists"); return saep
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sae=SAE(Z.shape[1],K,args.topk).to(dev); opt=torch.optim.Adam(sae.parameters(),lr=1e-3)
    Zt=torch.from_numpy(Z); n=Zt.shape[0]; bs=16384
    log(f"S1 train K={K} topk={args.topk} on {n:,} patches")
    for ep in range(args.epochs):
        perm=torch.randperm(n); tot=0.0
        for st in range(0,n,bs):
            zb=Zt[perm[st:st+bs]].to(dev,torch.float32)
            opt.zero_grad(); a,zh=sae(zb)
            loss=(1-F.cosine_similarity(zb,zh,dim=1)).mean()
            loss.backward(); opt.step(); sae._norm(); tot+=loss.item()*len(zb)
        if ep%10==0 or ep==args.epochs-1: log(f"  K={K} ep{ep:3d} recon={tot/n:.4f}")
    torch.save({"state":{k:v.cpu() for k,v in sae.state_dict().items()},
                "D":Z.shape[1],"K":K,"topk":args.topk,"res":args.res}, saep)
    log(f"S1 done K={K} -> {saep.name}")
    return saep

# ─────────────────────────── S2: street inference (all K in one pass) ────────
def infer_streets(ext, saeps, args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    saes={}                                              # K -> loaded SAE
    for p in saeps:
        d=torch.load(p,map_location="cpu"); m=SAE(d["D"],d["K"],d["topk"]).to(dev)
        m.load_state_dict(d["state"]); m.eval(); saes[d["K"]]=m
    Ks=sorted(saes); G=ext.G
    for city in args.cities:
        if all((OUT/f"streets_{city}_k{K}.npz").exists() for K in Ks):
            log(f"S2 skip {city} — all K exist"); continue
        tdir=OUT/"thumbs"/city; tdir.mkdir(parents=True,exist_ok=True)
        odp=REPO/"formal"/"ondisk"/f"{city}.parquet"
        if not odp.exists(): log(f"S2 {city}: missing {odp.name} (run find_ondisk_4city) — skip"); continue
        odf=pd.read_parquet(odp)                              # panos already on-disk 4-heading
        ppr=odf.groupby("matched_road_id").size().sort_values(ascending=False)
        roads=ppr[ppr>=3].index.tolist()[:args.street_roads]
        rows=[]
        for rid in roads:
            sub=odf[odf["matched_road_id"]==rid].sort_values("chainage_m")
            if len(sub)>args.street_pts:
                idx=np.linspace(0,len(sub)-1,args.street_pts).round().astype(int); sub=sub.iloc[idx]
            for _,r in sub.iterrows():
                rows.append((str(rid),r["pano_id"],float(r["lat"]),float(r["lon"]),float(r["chainage_m"])))
        if not rows: log(f"S2 {city}: no roads with >=3 on-disk panos"); continue
        plan=pd.DataFrame(rows,columns=["road","pano_id","lat","lon","chainage"])
        log(f"S2 {city}: {plan['road'].nunique()} roads, {len(plan)} points, extract @{args.res} + encode {Ks}")
        gmaps={K:np.full((len(plan),4,G,G),-1,np.int16) for K in Ks}
        for pi in range(len(plan)):
            pid=plan.iloc[pi]["pano_id"]
            pils,kept=load_pils([(city,pid,h) for h in HEADINGS],args.res)
            if len(pils)<4: continue
            ptf,art=ext.batch(pils)                          # (4,G*G,1024),(4,G*G) — extract ONCE
            pt=ptf.to(dev).reshape(-1,1024); art_flat=art.reshape(-1).numpy()
            for K in Ks:
                with torch.no_grad(): a=saes[K].encode(pt)
                gm=a.argmax(1).cpu().numpy().astype(np.int16)
                gm[art_flat]=-1                             # artifact patches -> no gene
                gmaps[K][pi]=gm.reshape(4,G,G)
            pils[0].resize((256,256)).save(tdir/f"{pi}.jpg",quality=72)
            if (pi+1)%50==0: log(f"  {city}: {pi+1}/{len(plan)}")
        for K in Ks:
            np.savez_compressed(OUT/f"streets_{city}_k{K}.npz", gmaps=gmaps[K],
                road=plan["road"].values, pano_id=plan["pano_id"].values,
                lat=plan["lat"].values, lon=plan["lon"].values, G=G)
        log(f"S2 {city} done -> streets_{city}_k*.npz ({len(plan)} points)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cities",nargs="+",default=list(CITY_DIR))
    ap.add_argument("--res",type=int,default=448)
    ap.add_argument("--dict-panos",type=int,default=4000)
    ap.add_argument("--keep-patches",type=int,default=200)
    ap.add_argument("--K-list",nargs="+",type=int,default=[256,512,1024])
    ap.add_argument("--topk",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--street-roads",type=int,default=60)
    ap.add_argument("--street-pts",type=int,default=5)
    ap.add_argument("--scan-roads",type=int,default=400)
    ap.add_argument("--art-factor",type=float,default=0.0,
                    help="drop patches whose raw norm > factor*mean (norm-based; DINOv3 artifacts are NOT high-norm, so leave 0)")
    ap.add_argument("--project-dirs",default="",
                    help="npy of artifact direction vectors to project OUT of features before SAE (root fix)")
    ap.add_argument("--infer-only",action="store_true",
                    help="skip S0/S1 training; load --sae-path and only run S2 prediction")
    ap.add_argument("--sae-path",default="",help="trained SAE to load for --infer-only")
    ap.add_argument("--no-thumbs",action="store_true",help="skip per-point thumbnails (large sweeps)")
    ap.add_argument("--sample-json",default="",help="json {city: n_panos} for per-city dict sampling")
    args=ap.parse_args()
    args.sample_map=json.load(open(args.sample_json)) if args.sample_json else None
    if args.sample_map: log(f"per-city sampling: {sum(args.sample_map.values()):,} panos over {len(args.sample_map)} cities")
    log(f"FORMAL RUN cities={args.cities} res={args.res} infer_only={args.infer_only} proj={args.project_dirs or 'none'}")
    if torch.cuda.is_available(): log("GPU:",torch.cuda.get_device_name(0))
    else: log("WARN: no GPU — running on CPU")
    ext=Extractor(args.res, art_factor=args.art_factor, proj_path=args.project_dirs or None)
    if args.infer_only:
        saeps=[args.sae_path]                             # predict-only with fixed trained dict
    else:
        Z=build_sample(ext,args)                          # one shared patch sample
        saeps=[train_sae(Z,K,args) for K in args.K_list]  # one SAE per K
        del Z
    infer_streets(ext,saeps,args)                         # extract streets, encode
    log("FORMAL RUN COMPLETE")

if __name__=="__main__":
    main()
