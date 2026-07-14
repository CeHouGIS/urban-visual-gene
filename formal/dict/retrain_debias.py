"""Retrain the global K=512 SAE with the FULL positional subspace projected out
(--project-dirs artifact_dirs_pos.npy). Reproduces the original de-biased sample
(3000 panos/city x 4 headings x 80 patches = 960k/city, 12 cities) but re-extracts
features with the new projection. Output dir set via FORMAL_OUT (use a fresh dir).
Skips street inference — only builds the sample and trains the dictionary."""
import os, json, types, time
from pathlib import Path
from formal.gpu_run import Extractor, build_sample, train_sae, OUT
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
POS=REPO/"formal"/"formal_out_global"/"artifact_dirs_pos.npy"
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo",
        "MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
assert POS.exists(), f"missing {POS} — run pos_subspace first"
smap=json.load(open(REPO/"formal"/"sample_counts.json"))
args=types.SimpleNamespace(cities=CITIES, res=448, dict_panos=3000, keep_patches=80,
    K_list=[512], topk=32, epochs=60, batch=32, art_factor=0.0,
    project_dirs=str(POS), sample_json=str(REPO/"formal"/"sample_counts.json"),
    sample_map=smap)
log(f"OUT={OUT}  project_dirs={args.project_dirs}")
log(f"cities={args.cities}")
ext=Extractor(args.res, art_factor=args.art_factor, proj_path=args.project_dirs)
Z=build_sample(ext, args)
log(f"sample ready: {Z.shape}")
saep=train_sae(Z, 512, args)
log(f"[done] trained -> {saep}")
