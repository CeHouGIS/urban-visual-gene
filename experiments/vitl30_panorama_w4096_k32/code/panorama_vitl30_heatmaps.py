"""Refresh evidence visualizations only; keep VLM inputs and numerical results fixed."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sae_experiments.panorama_vitl30 import OUTPUT, PATCHES
from sae_experiments.panorama_vitl30_analysis import checked_groups, draw_evidence, model_and_identity
from sae_experiments.panorama_vitl30_semantics import atomic_json, semantic_root
from sae_experiments.panorama_vitl30_figures import activation_examples


def main():
    torch.set_num_threads(8)
    root = semantic_root(OUTPUT)
    model, sha = model_and_identity(OUTPUT, 'cpu')
    groups = checked_groups(OUTPUT, sha)
    args = SimpleNamespace(output=OUTPUT, device='cpu', patch_batch=4*PATCHES,
                           evidence_autocast=False, evidence_root=root/'evidence')
    manifest = json.loads((root/'evidence_manifest.json').read_text())
    for group, examples in manifest['groups'].items():
        if all(e.get('render_version') == 'inferno_grey_v1' for e in examples):
            continue
        for index, old in enumerate(examples):
            rendered = draw_evidence(args, old, model, groups)
            examples[index] = {**old, **rendered, 'render_version': 'inferno_grey_v1'}
        atomic_json(root/'evidence_manifest.json', manifest)
        print('HIGH CONTRAST', group, len(examples), flush=True)
    # Refresh the six examples actually used in the main Chapter 5 figure.
    previous = json.loads((OUTPUT/'analysis/top_examples.json').read_text())
    args.evidence_root = OUTPUT/'analysis/evidence'
    used=set()
    for row in previous:
        if row['group'] in used:continue
        draw_evidence(args,row,model,groups);used.add(row['group'])
        if len(used)==6:break
    activation_examples(OUTPUT/'figures', OUTPUT)
    import shutil
    for suffix in ['.pdf','.png']:
        p=OUTPUT/'figures'/('Fig_ViTL30_Activation_Examples'+suffix)
        shutil.copy2(p,Path('/root/code/HKUST-Thesis/img_gene/vitl30_w4096')/p.name)
    print('HEATMAP REFRESH COMPLETE', flush=True)


if __name__=='__main__':main()
