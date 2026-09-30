"""Formal patch-level BatchTopK SAE run — GPU, 448 resolution.

Stages (each skipped if its output exists -> resumable on requeue):
  S0/S1  build a balanced patch sample from the selected cities @448, then
         train a shared BatchTopK SAE (the sparsity budget is batch-level).
  S2     for street-analysis panos (roads with on-disk 4-heading coverage),
         extract @448, encode with the SAE, save argmax gene maps + thumbnails
         + per-point metadata per city.

Downstream morphotype clustering + web assets are regenerated on the login node
from S2 outputs (CPU-only, no GPU needed).

  python -m formal.gpu_run --dict-panos 4000 --K-list 1024 --topk 8 --epochs 60 \
      --street-roads 60 --street-pts 5
"""
import os, sys, json, time, random, argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sae_experiments.models.base_sae import BatchTopKSAE
from sae_experiments.models.panorama_context import mix_panorama_context
from formal.quality.tiny_qc import QualityGate

MID   = os.environ.get("DINO_MODEL_PATH", "facebook/dinov3-vitl16-pretrain-lvd1689m")
_token_path = Path(os.environ.get("HF_TOKEN_FILE", Path.home()/".cache/huggingface/token"))
TOK   = os.environ.get("HF_TOKEN", _token_path.read_text().strip() if _token_path.exists() else "")
REPO  = Path(os.environ.get("UVG_REPO", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("GOOGLE_SV_ROOT", "/host/root/mnt/nas/huangyj/GoogleSV"))
IMROOT= DATA_ROOT / "images"
OUT   = Path(os.environ.get("FORMAL_OUT", REPO/"formal"/"formal_out_panorama_context")); OUT.mkdir(parents=True, exist_ok=True)
# Formal sample: 37 globally distributed cities with at least 12,800 complete
# four-heading panos verified in the NAS image catalogue. Experiment names are
# ASCII and stable; values are the image and metadata catalogue paths.
CITY_DIR = {
    "HongKong": "China/HongKong",
    "Taipei": "China/Taipei",
    "Singapore": "Singapore/Singapore",
    "Seoul": "Korea/Seoul",
    "Sapporo": "Japan/Sapporo",
    "Bengaluru": "India/Bengaluru",
    "Kolkata": "India/Kolkata",
    "Colombo": "SriLanka/Colombo",
    "Bandung": "Indonesia/Bandung",
    "Surabaya": "Indonesia/Surabaya",
    "Dubai": "UAE/Dubai",
    "Johannesburg": "SouthAfrica/Johannesburg",
    "CapeTown": "SouthAfrica/CapeTown",
    "Lagos": "Nigeria/Lagos",
    "Amsterdam": "Netherlands/Amsterdam",
    "Paris": "France/Paris",
    "Barcelona": "Spain/Barcelona",
    "Vienna": "Austria/Vienna",
    "Prague": "CzechRepublic/Prague",
    "Stockholm": "Sweden/Stockholm",
    "Oslo": "Norway/Oslo",
    "Moscow": "Russia/Moscow",
    "Lisboa": "Portugal/Lisboa",
    "BuenosAires": "Argentina/BuenosAires",
    "SaoPaulo": "Brazil/SaoPaulo",
    "RiodeJaneiro": "Brazil/RiodeJaneiro",
    "Bogota": "Colombia/Bogota",
    "NewYork": "US/NewYork",
    "LosAngeles": "US/LosAngeles",
    "Chicago": "US/Chicago",
    "SanFrancisco": "US/SanFrancisco",
    "Washington": "US/Washington",
    "Sydney": "Australia/Sydney",
    "Melbourne": "Australia/Melbourne",
    "Vancouver": "Canada/Vancouver",
    "Monterrey": "Mexico/Monterrey",
    "ChiangMai": "Thailand/ChiangMai",
}
META_CITY_DIR = CITY_DIR.copy()
HEADINGS=[0,90,180,270]

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
def imgpath(city,pid,h):
    # NAS catalogues shard by the first two pano-ID characters and preserve
    # their case (for example ``F/4/F4...``).  Keep a lower-case fallback for
    # older downloads that normalized the shard names.
    root = IMROOT / CITY_DIR[city]
    raw = root / pid[0] / pid[1] / f"{pid}_{h}.jpg"
    if raw.exists():
        return raw
    return root / pid[0].lower() / pid[1].lower() / f"{pid}_{h}.jpg"

# ─────────────────────────── DINOv3 batched extractor ───────────────────────
class _HiddenStateModel(nn.Module):
    """Return a tensor so DataParallel can gather transformer outputs safely."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values).last_hidden_state


class Extractor:
    def __init__(self, res, art_factor=0.0, proj_path=None, microbatch=32):
        from transformers import AutoImageProcessor, AutoModel
        self.res=res; self.G=res//16; self.art_factor=art_factor
        self.microbatch=max(1,int(microbatch))
        self.P=None
        if proj_path and Path(proj_path).exists():
            A=torch.tensor(np.load(proj_path),dtype=torch.float32)          # (m, D)
            Q,_=torch.linalg.qr(A.T); self.P=Q.T                            # orthonormal rows (m, D)
            log(f"projecting out {self.P.shape[0]} artifact directions from features")
        self.dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        load_kwargs = {}
        # ModelScope stores the authorized DINOv3 snapshot locally.  A local
        # path avoids another Hugging Face resolution/download attempt; the
        # token is only needed when MID remains a remote repository id.
        if not Path(MID).exists() and TOK:
            load_kwargs["token"] = TOK
        self.proc=AutoImageProcessor.from_pretrained(MID, **load_kwargs)
        backbone=AutoModel.from_pretrained(MID, **load_kwargs).to(self.dev).eval()
        if self.dev.type=="cuda": backbone=backbone.to(torch.bfloat16)
        # NCCL broadcast is unavailable on some multi-GPU login nodes.  Keep a
        # local frozen replica per device and split batches explicitly; this
        # preserves concurrent GPU use without relying on DataParallel's NCCL
        # communicator.
        self.models=[backbone]
        self._pool=None
        if self.dev.type=="cuda" and torch.cuda.device_count()>1:
            for device_index in range(1, torch.cuda.device_count()):
                replica=AutoModel.from_pretrained(MID, **load_kwargs).to(
                    torch.device(f"cuda:{device_index}")).eval().to(torch.bfloat16)
                self.models.append(replica)
            self._pool=ThreadPoolExecutor(max_workers=len(self.models))
            log(f"DINOv3 feature extraction across {len(self.models)} GPUs (manual split)")
        from PIL import Image
        with torch.no_grad():
            d=self.proc(images=Image.new("RGB",(res,res)),return_tensors="pt",
                        size={"height":res,"width":res})
            px=d["pixel_values"].to(self.dev)
            if self.dev.type=="cuda": px=px.to(torch.bfloat16)
            T=self.models[0](px).last_hidden_state.shape[1]
        self.prefix=T-self.G*self.G
        log(f"extractor res={res} grid={self.G}x{self.G} tokens={T} prefix={self.prefix} dev={self.dev}")

    @torch.no_grad()
    def batch(self, pils):
        """list[PIL] -> (feats (B,G*G,1024) L2-normed, artifact_mask (B,G*G) bool), cpu.
        Artifact = RAW (pre-normalize) token norm > art_factor * per-image mean norm
        (ViT register-artifact tokens are high-norm; detected here BEFORE normalization,
        which otherwise erases the norm signal). art_factor<=0 disables (mask all False)."""
        d=self.proc(images=pils,return_tensors="pt",size={"height":self.res,"width":self.res})
        if len(self.models)==1:
            px=d["pixel_values"].to(self.dev)
            if self.dev.type=="cuda": px=px.to(torch.bfloat16)
            h=self.models[0](px).last_hidden_state
        else:
            # Keep the decoded batch on host memory until each worker sends
            # only its shard to its assigned GPU.  Copying the whole batch to
            # cuda:0 first wastes memory and makes the second GPU wait.
            px=d["pixel_values"]
            chunks=torch.tensor_split(px,len(self.models),dim=0)
            def run_one(pair):
                device_index, (model, chunk) = pair
                if chunk.shape[0] == 0:
                    return None
                pieces=[]
                # The outer ``@torch.no_grad`` is thread-local; repeat it in
                # each worker or every microbatch retains a backward graph.
                with torch.no_grad():
                    start = 0
                    step = self.microbatch
                    while start < chunk.shape[0]:
                        # Keep the fast default batch, but recover from a
                        # transient activation OOM by retrying the same slice
                        # with a smaller microbatch.  This is local to the
                        # worker GPU and does not invalidate completed pieces.
                        width = min(step, chunk.shape[0] - start)
                        try:
                            piece=chunk[start:start+width].to(
                                f"cuda:{device_index}", dtype=torch.bfloat16
                            )
                            pieces.append(model(piece).last_hidden_state.to(self.dev))
                            start += width
                        except RuntimeError as exc:
                            if "out of memory" not in str(exc).lower() or step <= 1:
                                raise
                            del piece
                            torch.cuda.empty_cache()
                            step=max(1, step // 2)
                            log(f"GPU {device_index}: activation OOM; retrying with microbatch {step}")
                return torch.cat(pieces,dim=0)
            outputs=self._pool.map(run_one, enumerate(zip(self.models,chunks)))
            outputs=[x for x in outputs if x is not None]
            h=torch.cat(outputs,dim=0)
        h=h[:,self.prefix:,:].float()                                      # (B, G*G, 1024) RAW
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


def load_panorama_pils(city, pano, res):
    """Load all four views of one panorama, or return ``None`` if incomplete."""
    from PIL import Image

    views = []
    try:
        for heading in HEADINGS:
            views.append(Image.open(imgpath(city, pano, heading)).convert("RGB").resize(
                (res, res), Image.BILINEAR
            ))
    except Exception:
        return None
    return views


def load_panorama_batch(city, panos, res, workers=16):
    """Read a batch of complete panoramas concurrently from the NAS."""
    workers=max(1, min(int(workers), len(panos) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded=list(pool.map(lambda pid: load_panorama_pils(city, pid, res), panos))
    return [(pid, views) for pid, views in zip(panos, loaded) if views is not None]


def contextualize_views(features, artifact_mask, grid_size, context_weight):
    """Apply the fixed ring-context filter while preserving the four-view layout."""
    n_views, n_patches, dim = features.shape
    if n_views % len(HEADINGS) != 0:
        raise ValueError(f"expected a multiple of four views, got {n_views}")
    n_panos = n_views // len(HEADINGS)
    grouped = features.reshape(n_panos, len(HEADINGS), n_patches, dim)
    valid = ~artifact_mask.reshape(n_panos, len(HEADINGS), n_patches)
    contextual = mix_panorama_context(
        grouped,
        grid_size=grid_size,
        context_weight=context_weight,
        valid_mask=valid,
    )
    return contextual, valid

# ─────────────────────────── BatchTopK SAE ─────────────────────────────────
class SAE(BatchTopKSAE):
    """Formal name for the shared BatchTopK implementation.

    The aliases preserve the historical formal scripts' API. Older formal
    checkpoints with ``enc``/``dec`` state-dict names remain loadable.
    """
    def __init__(self, D, K, topk):
        super().__init__(input_dim=D, latent_dim=K, k=topk, decoder_unit_norm=True)

    @property
    def D(self): return self.input_dim
    @property
    def K(self): return self.latent_dim
    @property
    def topk(self): return self.k
    @property
    def enc(self): return self.encoder
    @property
    def dec(self): return self.decoder

    def _norm(self):
        self.normalize_decoder_()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        translated = {
            key.replace("enc.", "encoder.").replace("dec.", "decoder."): value
            for key, value in state_dict.items()
        }
        return super().load_state_dict(translated, strict=strict, assign=assign)

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
    """Collect complete four-heading pano IDs from the image tree.

    This is the fallback when a city has no SQLite metadata DB.  A heading-0
    file alone is insufficient for the panorama-context method, so candidates
    are retained only when all four requested images are present.
    """
    root=IMROOT/CITY_DIR[city]; ids=[]
    for r,dirs,files in os.walk(root):
        dirs.sort()
        for f in files:
            if not f.endswith("_0.jpg"):
                continue
            pid=f[:-6]
            if all((Path(r)/f"{pid}_{h}.jpg").is_file() for h in HEADINGS):
                ids.append(pid)
        if len(ids)>=n*stride: break
    rng.shuffle(ids); return ids[:n]

def _db(city, sub, fname):
    country, meta_city = META_CITY_DIR[city].split("/")
    return str(DATA_ROOT / "metadata" / country / meta_city / sub /
               fname.format(country=country, city=meta_city))
def meta_db(city): return _db(city,"meta","{country}_{city}.db")
def points_db(city): return _db(city,"sampling_points_db","combined.db")

def bad_panos(city):
    """QC blocklist: pano_ids flagged black/blur/tunnel by quality_filter -> qc/<city>.parquet.
    Empty set if no QC has been run for the city (so sampling is unaffected until then)."""
    if not hasattr(bad_panos,"_c"): bad_panos._c={}
    if city not in bad_panos._c:
        ff=REPO/"formal"/"qc_full"/f"{city}.parquet"          # full-sample CNN blocklist (preferred)
        f=ff if ff.exists() else REPO/"formal"/"qc"/f"{city}.parquet"; s=set()
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
    country, meta_city = META_CITY_DIR[city].split("/")
    pdir=str(DATA_ROOT / "metadata" / country / meta_city / "sampling_points_db")
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
    qc_model = getattr(args, "qc_model", "")
    qc = None if getattr(args, "no_qc", False) else QualityGate(qc_model or None)
    if qc is not None:
        log(f"S0 quality gate: {'learned '+str(qc.model_path) if qc.model_loaded else 'rule fallback'}")
    log(
        "S0 collecting contextual dictionary patch sample (complete four-view "
        f"panoramas), per-city={'sample_counts.json' if smap else args.dict_panos}, "
        f"context_weight={args.context_weight} ..."
    )
    for city in args.cities:
        cf=OUT/f"dict_{city}.f16.npy"
        if cf.exists():                                      # per-city checkpoint (resume on requeue)
            log(f"  {city}: cached {np.load(cf,mmap_mode='r').shape[0]:,} patches — skip"); continue
        per=smap.get(city, args.dict_panos)
        cands=stratified_panos(city, per, seed=0)            # ~6x oversample (weighted)
        ondisk=[]                                            # Phase 1: verify complete four-view coverage
        for pid in cands:
            if all(imgpath(city, pid, heading).is_file() for heading in HEADINGS):
                ondisk.append(pid)
            if len(ondisk)>=per: break
        log(f"  {city}: {len(ondisk):,}/{per:,} complete four-view panos from {len(cands):,} candidates")
        if len(ondisk) < per:
            log(
                f"  WARNING {city}: only {len(ondisk):,} complete panos available; "
                "the formal quota cannot be met from the current NAS catalogue"
            )
        # Persist each panorama batch separately.  A killed job can resume at
        # the next batch without retaining the entire city's patch sample in
        # RAM or repeating completed DINOv3 work.
        pano_batch_size = max(1, args.batch // len(HEADINGS))
        # Include the batch and patch-sampling parameters in the directory so
        # changing throughput settings cannot accidentally reuse incompatible
        # chunk boundaries from an older run.
        chunk_dir=OUT/"dict_chunks"/city/f"b{pano_batch_size}_p{args.keep_patches}"
        chunk_dir.mkdir(parents=True,exist_ok=True)
        qc_stats = {"candidates": 0, "black": 0, "tunnel": 0, "kept": 0}
        for batch_no, i in enumerate(range(0, len(ondisk), pano_batch_size)):
            chunk_path=chunk_dir/f"{batch_no:06d}.f16.npy"
            if chunk_path.exists():
                log(f"  {city}: batch {batch_no} cached — skip")
                continue
            pano_batch = ondisk[i:i + pano_batch_size]
            # The gate downsamples in memory, so each view is read from NAS
            # only once.  Rejected panoramas never reach the DINOv3 batch.
            loaded_qc = load_panorama_batch(city, pano_batch, args.res, args.io_workers)
            if qc is not None:
                loaded_qc, qst = qc.filter_loaded(loaded_qc)
                for key in qc_stats:
                    qc_stats[key] += qst[key]
            loaded = loaded_qc
            pano_pils = []
            kept_panos = []
            for pid, views in loaded:
                pano_pils.extend(views)
                kept_panos.append(pid)
            if not pano_pils:
                continue
            pt, art = ext.batch(pano_pils)
            contextual, valid = contextualize_views(
                pt, art, ext.G, args.context_weight
            )
            contextual = contextual.numpy()
            valid = valid.numpy()
            batch_buf=[]
            for r in range(contextual.shape[0]):
                for direction in range(len(HEADINGS)):
                    ok = np.flatnonzero(valid[r, direction])
                    if len(ok) == 0:
                        continue
                    sel = rng.sample(list(ok), min(args.keep_patches, len(ok)))
                    batch_buf.append(contextual[r, direction, sel].astype(np.float16))
            if batch_buf:
                cz=np.concatenate(batch_buf,0)
                tmp=chunk_path.with_suffix(".tmp.npy")
                np.save(tmp,cz)
                os.replace(tmp,chunk_path)
            if batch_no % 10 == 0:
                log(f"  {city}: {i:,}/{len(ondisk):,} panos")
        if qc is not None:
            log(f"  {city}: QC candidates={qc_stats['candidates']:,} black={qc_stats['black']:,} "
                f"tunnel={qc_stats['tunnel']:,} kept={qc_stats['kept']:,}")
        chunks=sorted(chunk_dir.glob("*.f16.npy"))
        if chunks:
            cz=np.concatenate([np.load(p) for p in chunks],0)
            tmp=cf.with_suffix(".tmp.npy")
            np.save(tmp,cz)
            os.replace(tmp,cf)
            log(f"  {city}: {cz.shape[0]:,} patches -> {cf.name}")
    parts=[np.load(OUT/f"dict_{c}.f16.npy") for c in args.cities if (OUT/f"dict_{c}.f16.npy").exists()]
    Z=np.concatenate(parts,0); np.save(sp,Z)
    log(f"S0 done: {Z.shape[0]:,} patches x {Z.shape[1]} -> {sp.name}")
    return Z

# ─────────────────────────── S1: train one SAE per K ────────────────────────
def train_sae(Z, K, args):
    saep=OUT/f"sae_{args.res}_k{K}.pt"
    if saep.exists(): log(f"S1 skip K={K} — exists"); return saep
    resumep=OUT/f"sae_{args.res}_k{K}.train.pt"
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sae=SAE(Z.shape[1],K,args.topk).to(dev); opt=torch.optim.Adam(sae.parameters(),lr=1e-3)
    Zt=torch.from_numpy(Z); n=Zt.shape[0]; bs=16384
    start_epoch=0
    if resumep.exists():
        saved=torch.load(resumep,map_location=dev,weights_only=False)
        saved_shape=(saved["D"],saved["K"],saved["topk"],saved["res"],saved["n"])
        if saved_shape != (Z.shape[1],K,args.topk,args.res,n):
            raise ValueError(f"incompatible training checkpoint: {resumep}")
        sae.load_state_dict(saved["state"])
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if dev=="cuda" and "rng_cuda" in saved:
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        start_epoch=saved["epoch"]+1
        log(f"S1 resume K={K} at epoch {start_epoch}/{args.epochs}")
    log(f"S1 train K={K} topk={args.topk} on {n:,} patches, epoch {start_epoch}/{args.epochs}")
    for ep in range(start_epoch,args.epochs):
        perm=torch.randperm(n); tot=0.0
        for st in range(0,n,bs):
            zb=Zt[perm[st:st+bs]].to(dev,torch.float32)
            opt.zero_grad(); a,zh=sae(zb)
            loss=(1-F.cosine_similarity(zb,zh,dim=1)).mean()
            loss.backward(); opt.step(); sae._norm(); tot+=loss.item()*len(zb)
        if ep%10==0 or ep==args.epochs-1: log(f"  K={K} ep{ep:3d} recon={tot/n:.4f}")
        training_state={"state":sae.state_dict(),"optimizer":opt.state_dict(),
                        "epoch":ep,"D":Z.shape[1],"K":K,"topk":args.topk,
                        "res":args.res,"n":n,"rng_cpu":torch.get_rng_state()}
        if dev=="cuda":
            training_state["rng_cuda"]=torch.cuda.get_rng_state_all()
        tmp=resumep.with_suffix(".tmp")
        torch.save(training_state,tmp)
        os.replace(tmp,resumep)
    final={"state":{k:v.cpu() for k,v in sae.state_dict().items()},
                "D":Z.shape[1],"K":K,"topk":args.topk,"res":args.res,
                "method":"panorama_context_batchtopk",
                "context_weight":args.context_weight,
                # train_sae is intentionally independent of the extractor;
                # the patch grid is determined by the configured resolution.
                "context_grid":args.res // 16,
                "context_layout":"4 directions laid out on a horizontal circular ring"}
    tmp=saep.with_suffix(".tmp")
    torch.save(final,tmp)
    os.replace(tmp,saep)
    log(f"S1 done K={K} -> {saep.name}")
    return saep

# ─────────────────────────── S2: street inference (all K in one pass) ────────
def infer_streets(ext, saeps, args):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    saes={}                                              # K -> loaded SAE
    context_weight = args.context_weight
    for p in saeps:
        d=torch.load(p,map_location="cpu"); m=SAE(d["D"],d["K"],d["topk"]).to(dev)
        m.load_state_dict(d["state"]); m.eval(); saes[d["K"]]=m
        context_weight = float(d.get("context_weight", context_weight))
    log(f"S2 panorama context weight={context_weight}")
    qc_model = getattr(args, "qc_model", "")
    qc = None if getattr(args, "no_qc", False) else QualityGate(qc_model or None)
    if qc is not None:
        log(f"S2 quality gate: {'learned '+str(qc.model_path) if qc.model_loaded else 'rule fallback'}")
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
            pils=load_panorama_pils(city, pid, args.res)
            if qc is not None and pils is not None and qc.score_views(pils)["is_bad"]:
                continue
            if pils is None: continue
            ptf,art=ext.batch(pils)                          # extract all four views once
            contextual, valid = contextualize_views(
                ptf, art, ext.G, context_weight
            )
            pt=contextual.to(dev).reshape(-1,1024)
            valid_flat=valid.reshape(-1).numpy()
            for K in Ks:
                with torch.no_grad(): a=saes[K].encode(pt)
                gm=a.argmax(1).cpu().numpy().astype(np.int16)
                gm[~valid_flat]=-1                          # invalid/artifact centers -> no gene
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
    ap.add_argument("--keep-patches",type=int,default=32,
                    help="random valid patches retained per view and panorama; 32 keeps the full-city cache bounded")
    ap.add_argument("--K-list",nargs="+",type=int,default=[1024])
    ap.add_argument("--topk",type=int,default=8)
    ap.add_argument("--context-weight",type=float,default=0.25,
                    help="neighbor contribution in the 4-view circular context filter")
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--batch",type=int,default=256,
                    help="number of direction images per DINOv3 batch; split across GPUs and adapt on OOM")
    ap.add_argument("--microbatch",type=int,default=64,
                    help="per-GPU DINOv3 forward size; halves automatically if activation memory is tight")
    ap.add_argument("--io-workers",type=int,default=32,
                    help="parallel NAS image readers per panorama batch")
    ap.add_argument("--street-roads",type=int,default=60)
    ap.add_argument("--street-pts",type=int,default=5)
    ap.add_argument("--scan-roads",type=int,default=400)
    ap.add_argument("--art-factor",type=float,default=0.0,
                    help="drop patches whose raw norm > factor*mean (norm-based; DINOv3 artifacts are NOT high-norm, so leave 0)")
    ap.add_argument("--project-dirs",default="",
                    help="npy of artifact direction vectors to project OUT of features before SAE (root fix)")
    ap.add_argument("--infer-only",action="store_true",
                    help="skip S0/S1 training; load --sae-path and only run S2 prediction")
    ap.add_argument("--skip-infer",action="store_true",
                    help="train/cache the dictionary but do not run street-point S2 inference")
    ap.add_argument("--train-only",action="store_true",
                    help="train SAE from the existing dictionary sample without loading DINOv3")
    ap.add_argument("--sae-path",default="",help="trained SAE to load for --infer-only")
    ap.add_argument("--no-thumbs",action="store_true",help="skip per-point thumbnails (large sweeps)")
    ap.add_argument("--sample-json",default="",help="json {city: n_panos} for per-city dict sampling")
    ap.add_argument("--qc-model",default="",help="joblib tiny QC checkpoint; defaults to formal/quality/qc_model.joblib")
    ap.add_argument("--no-qc",action="store_true",help="disable the pre-DINO black/tunnel quality gate")
    args=ap.parse_args()
    # Keep resumed runs and checkpoint metadata reproducible across retries.
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args.sample_map=json.load(open(args.sample_json)) if args.sample_json else None
    if args.sample_map: log(f"per-city sampling: {sum(args.sample_map.values()):,} panos over {len(args.sample_map)} cities")
    log(f"FORMAL RUN cities={args.cities} res={args.res} infer_only={args.infer_only} proj={args.project_dirs or 'none'}")
    if torch.cuda.is_available(): log("GPU:",torch.cuda.get_device_name(0))
    else: log("WARN: no GPU — running on CPU")
    if args.train_only and args.infer_only:
        ap.error("--train-only and --infer-only cannot be combined")
    if args.train_only and not args.skip_infer:
        ap.error("--train-only requires --skip-infer")
    ext=None if args.train_only else Extractor(
        args.res, art_factor=args.art_factor, proj_path=args.project_dirs or None,
        microbatch=args.microbatch)
    if args.infer_only:
        saeps=[args.sae_path]                             # predict-only with fixed trained dict
    else:
        if args.train_only:
            sample=OUT/"dict_sample.f16.npy"
            if not sample.is_file():
                raise FileNotFoundError(f"dictionary sample missing: {sample}")
            Z=np.load(sample,mmap_mode="r")
            log(f"S0 reuse sample {Z.shape} (memory mapped)")
        else:
            Z=build_sample(ext,args)                      # one shared patch sample
        saeps=[train_sae(Z,K,args) for K in args.K_list]  # one SAE per K
        del Z
    if not args.skip_infer:
        infer_streets(ext,saeps,args)                     # extract streets, encode
    else:
        log("S2 skipped by --skip-infer")
    log("FORMAL RUN COMPLETE")

if __name__=="__main__":
    main()
