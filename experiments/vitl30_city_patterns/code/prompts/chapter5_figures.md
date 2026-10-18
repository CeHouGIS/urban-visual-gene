Use the verified 30-city grid-sample analysis to refresh Chapter 5 of
/root/code/HKUST-Thesis and the current Urban Scene Primitives website. Read
the current working trees and preserve all existing user changes. The main
SAE training cohort and its frozen checkpoint are distinct from the new
688,174-candidate spatial sample; training-loss and decoder-hierarchy plots
must retain their original denominators and provenance. Never imply the SAE
was retrained on the new sample.

Audit every figure included from 05_research3.tex,
chapter5_vitl30_figures.tex and chapter5_vitl30_semantics.tex. Regenerate
every data-dependent city, composition, similarity, co-occurrence, spatial,
UMAP, activation and interpretation figure from the appropriate current
result. Add figure(s) showing each defensible landscape prototype's
distribution across cities and within 1-km cells, and four-direction original
and masked examples distinguishing same-side, opposite-side and undetermined
cases. Recheck pipeline, panorama-aware, training-loss and dictionary figures
without changing their training identity. Update captions, table denominators,
methods, results and limitations. Do not substitute normalized group mass
for dominant-patch prevalence without explicitly changing the statistic.

Implement reproducible plotting code, run relevant tests and compile the
thesis. Produce figure_inventory.json listing at least the eleven existing
Fig_ViTL30_* figures with old/new provenance and paths; compile_report.json
with passed=true only if the compiler exits successfully and the PDF has
the expected pages; and copy the compiled PDF as thesis_preview.pdf into the
stage result directory. Publish site and GitHub changes only after checking
the complete result and restricting the commit to files you intentionally
changed. Do not describe provisional Qwen labels as human-validated truth.
