# Urban Visual Gene paper draft

Compile from this directory:

```bash
pdflatex urban_visual_gene_draft.tex
bibtex urban_visual_gene_draft
pdflatex urban_visual_gene_draft.tex
pdflatex urban_visual_gene_draft.tex
```

Red `TODO` markers identify author metadata, collection details, ethics/licensing,
and release information that cannot be inferred safely from the current artifacts.

The draft deliberately distinguishes completed evidence from pending work. In
particular, Qwen labels are reported as preliminary annotations, not validated
semantic fidelity scores.

## Atypical visual co-occurrence

The paper package also includes the new E+D+P-distance-weighted co-occurrence
analysis. Its frozen data are under `data/atypical_cooccurrence/`, Tables 6--7
summarize the leading global and city-specific combinations, and the two
associated figures are stored in `figures/main/`.

The strict same-image extension is stored under `data/image_level_atypicality/`.
It requires both elements to win at least four of 196 patches in the same
directional image; Table 8 and `Fig_Image_Level_Atypical_Coactivation` report
the resulting pairs and representative activation overlays.
