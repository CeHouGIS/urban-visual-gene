"""Compose the two existing Dhaka case figures for Chapter 5 without inference."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'experiments/vitl30_city_patterns/results/city_internal_compositions'
ATLAS = RESULTS / 'dhaka_local_atlas'
EXAMPLES = RESULTS / 'examples'
SCOPE_FIGURE = RESULTS / 'dhaka_three_scopes/Fig_Dhaka_Three_Scopes.pdf'
OUTPUT = RESULTS / 'dhaka_thesis_figures'
THESIS = Path('/root/code/HKUST-Thesis/img_gene/vitl30_w4096')
LOCAL_STEM = 'Fig_ViTL30_Dhaka_Local_Composition_Examples'
SCOPE_STEM = 'Fig_ViTL30_Dhaka_Three_Scopes'
LOCAL_PATTERN = 'cd8d335c382b'
EXAMPLE_RANKS = (1, 2, 4, 5)


def compose_local_figure(manifest: dict, target: Path, preview: Path):
    pattern = next(row for row in manifest['patterns']
                   if row['pattern_id'] == LOCAL_PATTERN)
    map_and_example = Path(pattern['figure'])
    panels = [EXAMPLES / f"Dhaka_{LOCAL_PATTERN}_{rank}.png"
              for rank in EXAMPLE_RANKS]
    if not map_and_example.is_file() or any(not path.is_file() for path in panels):
        raise FileNotFoundError('Existing Dhaka activation and example figures are incomplete')

    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(12, 13), facecolor='white')
    grid = fig.add_gridspec(3, 2, height_ratios=(1.34, 1, 1),
                            left=.008, right=.992, top=.992, bottom=.008,
                            wspace=.018, hspace=.035)
    axes = [fig.add_subplot(grid[0, :])]
    axes.extend((fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
                 fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])))
    images = [map_and_example, *panels]
    for axis, path in zip(axes, images):
        with Image.open(path) as opened:
            axis.imshow(opened.convert('RGB'), interpolation='lanczos')
        axis.set_axis_off()

    fig.savefig(target, facecolor='white', dpi=300)
    fig.savefig(preview, facecolor='white', dpi=220)
    plt.close(fig)
    return pattern


def main():
    manifest_path = ATLAS / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('city') != 'Bangladesh/Dhaka'
            or manifest.get('sampled_panoramas') != 93400
            or manifest.get('observed_cells') != 9587):
        raise ValueError('Unexpected Dhaka atlas identity or denominator')
    if not SCOPE_FIGURE.is_file():
        raise FileNotFoundError(SCOPE_FIGURE)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    THESIS.mkdir(parents=True, exist_ok=True)
    local_pdf = OUTPUT / f'{LOCAL_STEM}.pdf'
    local_png = OUTPUT / f'{LOCAL_STEM}.png'
    local_pattern = compose_local_figure(manifest, local_pdf, local_png)

    scope_pdf = OUTPUT / f'{SCOPE_STEM}.pdf'
    shutil.copy2(SCOPE_FIGURE, scope_pdf)
    shutil.copy2(local_pdf, THESIS / local_pdf.name)
    shutil.copy2(scope_pdf, THESIS / scope_pdf.name)

    value = {
        'status': 'complete',
        'city': manifest['city'],
        'sampled_panoramas': manifest['sampled_panoramas'],
        'observed_cells': manifest['observed_cells'],
        'local_combination': {
            'pattern_id': local_pattern['pattern_id'],
            'label': local_pattern['english'],
            'panoramas': local_pattern['panoramas'],
            'cells': local_pattern['cells'],
            'cell_balanced_rate': local_pattern['city_rate'],
            'instances': [
                str(EXAMPLES / f'Dhaka_{LOCAL_PATTERN}_{rank}.png')
                for rank in (3, *EXAMPLE_RANKS)
            ],
            'figure': local_pdf.name,
        },
        'scope_comparison': {
            'source': str(SCOPE_FIGURE),
            'figure': scope_pdf.name,
        },
        'inference_rerun': False,
        'interpretation': 'Existing frozen-SAE response overlays are relative attribution, not segmentation. Dhaka-only recurrence does not establish geographic exclusivity.',
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {local_pdf}')
    print(f'Installed {THESIS / local_pdf.name}')
    print(f'Installed {THESIS / scope_pdf.name}')


if __name__ == '__main__':
    main()
