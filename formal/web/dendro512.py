"""Regenerate the 512-atom phylogeny PNG (interpret.html) from a chosen dict.
CPU, matplotlib (no CJK needed). Reads DICTDIR/sae_448_k512.pt."""
import os, numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from scipy.spatial.distance import squareform
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
DICTDIR=REPO/"formal"/os.environ.get("DICTDIR_NAME","formal_out_global2")
GAL=DICTDIR/"web"/"gallery"; GAL.mkdir(parents=True,exist_ok=True)
d=torch.load(DICTDIR/"sae_448_k512.pt",map_location="cpu")
A=d["state"]["dec.weight"].T.numpy(); A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
D=1-np.clip(A@A.T,-1,1); np.fill_diagonal(D,0); D=(D+D.T)/2
Z=linkage(squareform(D,checks=False),method="average")
set_link_color_palette(["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff","#f6c945","#9b8cff","#ff9e64","#73daca"])
fig,ax=plt.subplots(figsize=(14,5.5)); fig.patch.set_facecolor("#070b14"); ax.set_facecolor("#070b14")
dendrogram(Z,no_labels=True,color_threshold=Z[-11,2],above_threshold_color="#8a9bb5",ax=ax)
ax.set_title("512 visual-gene phylogeny (de-biased dict, cosine distance)",color="#dbe4f0",fontsize=13)
ax.tick_params(colors="#8a9bb5"); [s.set_visible(False) for s in ax.spines.values()]
fig.tight_layout(); fig.savefig(GAL/"dendrogram_512.png",dpi=120,facecolor="#070b14"); plt.close(fig)
print("saved dendrogram_512.png ->",GAL)
