"""Map every Dhaka-only confirmed combination and render one four-view example each."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

from .composition_evidence import COLORS, infer, render
from .composition_patterns import pattern_mask
from .extract import EXPECTED_CHECKPOINT, image_paths, load_models, write_json
from .followup import FOLLOWUP


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/city_internal_compositions'
OUTPUT = RESULTS / 'dhaka_local_atlas'
STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
FRAME = FOLLOWUP / 'stages/candidate_combinations/panorama_elements.parquet'
CITY = 'Bangladesh/Dhaka'
ORDER_COLORS = {2: '#4e817d', 3: '#b07854', 4: '#80638f'}
LABELS = {
    '围墙': 'wall', '围栏': 'fence', '广告牌': 'signs/posters',
    '树木': 'trees', '水体': 'water', '沙地': 'sandy ground',
    '灌木': 'shrubs', '电线': 'wires', '电线杆': 'utility pole',
    '管道': 'pipes', '老屋': 'older structures', '草地': 'grass',
    '路灯': 'streetlight', '金属屋顶建筑': 'metal-roof building',
}
PIL_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def english_label(label: str) -> str:
    parts = [part.strip() for part in label.split('＋')]
    missing = [part for part in parts if part not in LABELS]
    if missing:
        raise ValueError(f'Untranslated semantic labels: {missing}')
    return ' + '.join(LABELS[part] for part in parts)


def load_inputs(stage: Path, frame_path: Path, results: Path):
    rows = pd.read_csv(results / 'city_patterns.csv')
    local = rows[(rows.city == CITY) & rows.replicated &
                 (rows.cities_replicated == 1)].copy()
    local = local.sort_values(['order', 'city_rate', 'pattern_id'],
                              ascending=[False, False, True]).reset_index(drop=True)
    if len(local) != 55 or local.pattern_id.duplicated().any():
        raise ValueError(f'Expected 55 distinct Dhaka-only candidates, got {len(local)}')
    frame = pd.read_parquet(frame_path, columns=['city', 'panoid', 'grid_x', 'grid_y'])
    assignment = np.load(stage / 'assignment.npy', mmap_mode='r')
    scores = np.load(stage / 'scores.npy', mmap_mode='r')
    units = json.loads((stage / 'units.json').read_text())
    calibration = json.loads((stage / 'calibration.json').read_text())
    if len(frame) != len(assignment) or len(frame) != len(scores):
        raise ValueError('Metadata and response arrays are misaligned')
    city_positions = np.flatnonzero(frame.city.to_numpy() == CITY)
    if not len(city_positions):
        raise ValueError('Dhaka is absent from the panorama corpus')
    city_frame = frame.iloc[city_positions].reset_index(drop=True)
    cell_index, cells = pd.factorize(
        pd.MultiIndex.from_frame(city_frame[['grid_x', 'grid_y']]), sort=True)
    cell_frame = cells.to_frame(index=False)
    cell_frame.columns = ['grid_x', 'grid_y']
    cell_frame['sampled_panoramas'] = np.bincount(cell_index, minlength=len(cells))
    if len(city_frame) != 93400 or len(cell_frame) != 9587:
        raise ValueError('Dhaka corpus denominator changed')
    thresholds = np.asarray(calibration['unit_thresholds'], dtype=float)
    return local, frame, assignment, scores, units, thresholds, city_positions, \
        city_frame, cell_index, cell_frame


def select_example(frame, scores, thresholds, city_positions, positive, ids,
                   used_panoids, minimum_rank):
    positions = city_positions[np.flatnonzero(positive)]
    ratios = scores[positions[:, None], np.asarray(ids)[None, :]] / \
        thresholds[np.asarray(ids)][None, :]
    strength = np.min(ratios, axis=1)
    order = np.argsort(-strength, kind='stable')
    for rank, order_index in enumerate(order, 1):
        if rank < minimum_rank:
            continue
        position = int(positions[order_index])
        sample = frame.iloc[position]
        panoid = str(sample.panoid)
        if panoid in used_panoids:
            continue
        if all(path.is_file() for path in image_paths(CITY, panoid)):
            used_panoids.add(panoid)
            return sample, rank, float(strength[order_index])
    raise ValueError('No distinct complete four-view example found')


def draw_city_map(ax, cells, selected, sample=None, title=None):
    xmin, xmax = int(cells.grid_x.min()) - 3, int(cells.grid_x.max()) + 3
    ymin, ymax = int(cells.grid_y.min()) - 3, int(cells.grid_y.max()) + 3
    ax.scatter(cells.grid_x, cells.grid_y, s=.6, marker='s', linewidths=0,
               color='#dce1df', rasterized=True)
    ax.scatter(selected.grid_x, selected.grid_y, s=2.2, marker='s', linewidths=0,
               color=selected.attrs.get('color', '#866a94'), rasterized=True)
    if sample is not None:
        ax.scatter([sample.grid_x], [sample.grid_y], s=68, marker='o',
                   facecolor='none', edgecolor='#17282a', linewidth=1.1, zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc='left', fontsize=10, color='#263432', pad=8)
    ax.plot([xmin + 6, xmin + 16], [ymin + 7, ymin + 7],
            lw=1.5, color='#263432')
    ax.text(xmin + 11, ymin + 10, '10 km', ha='center', fontsize=8,
            color='#263432')


def draw_page(record, cells, counts, folder: Path, target: Path, pdf: PdfPages):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(14.5, 6.2), facecolor='white')
    grid = fig.add_gridspec(2, 2, width_ratios=[4.0, 8.3],
                           left=.045, right=.985, top=.78, bottom=.11,
                           wspace=.1, hspace=.13)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_original = fig.add_subplot(grid[0, 1])
    ax_overlay = fig.add_subplot(grid[1, 1])
    active = cells.loc[counts > 0, ['grid_x', 'grid_y']].copy()
    active.attrs['color'] = ORDER_COLORS[record['order']]
    draw_city_map(ax_map, cells, active, record['sample'],
                  f"1-km cells | {record['cells']:,} of {len(cells):,} observed")
    for ax, filename, heading in ((ax_original, 'original.webp', 'Four-view original'),
                                  (ax_overlay, 'overlay.webp', 'SAE response overlay')):
        with Image.open(folder / filename) as opened:
            ax.imshow(opened.convert('RGB'), aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(heading, loc='left', fontsize=10, color='#263432', pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.045, .946, f"{record['index']:02d}  {record['english']}",
             fontsize=16, fontweight='bold', color='#24312f', ha='left')
    fig.text(.045, .890,
             f"Dhaka-only confirmed candidate  |  {record['order']} elements  |  "
             f"{record['panoramas']:,} sampled panoramas  |  "
             f"{record['city_rate']:.2%} cell-balanced response",
             fontsize=9.3, color='#52615f', ha='left')
    fig.text(.045, .84,
             f"Pattern {record['pattern_id']}  |  "
             f"Illustrated response rank {record['strength_rank']}  |  "
             f"Panoid {record['panoid']}",
             fontsize=8.3, color='#52615f', ha='left')
    x = .39
    for raw, color in zip(record['unit_labels'], COLORS):
        label = LABELS[raw]
        fig.text(x, .065, '\u25a0', color=color, fontsize=11, va='center')
        fig.text(x + .014, .065, label, color='#455250', fontsize=8.4, va='center')
        x += .035 + .0073 * len(label)
    fig.text(.045, .035,
             'A colored cell has at least one matching sampled panorama. '
             'Response labels are provisional; colored pixels are not object segmentation.',
             fontsize=8.2, color='#586460', ha='left')
    fig.savefig(target, dpi=175, facecolor='white')
    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def draw_overview(cells: pd.DataFrame, output: Path, selected_count: int,
                  matching_panoramas: int):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.2), facecolor='white')
    fig.subplots_adjust(left=.05, right=.92, top=.82, bottom=.13, wspace=.16)
    positive = cells.candidate_types > 0
    ax = axes[0]
    ax.scatter(cells.grid_x, cells.grid_y, s=1.2, marker='s', linewidths=0,
               color='#dce1df', rasterized=True)
    ax.scatter(cells.loc[positive, 'grid_x'], cells.loc[positive, 'grid_y'],
               s=2.2, marker='s', linewidths=0, color='#4e817d', rasterized=True)
    ax.set_title(f"a  At least one match: {int(positive.sum()):,}/{len(cells):,} cells",
                 loc='left', fontsize=11, fontweight='bold', color='#263432')
    ax = axes[1]
    upper = max(22, selected_count + 1)
    bounds = [0, 1, 3, 6, 11, 21, upper]
    palette = ListedColormap(['#e0e5e3', '#d6e6e2', '#99c7bc',
                              '#519687', '#bb8761', '#765e86'])
    norm = BoundaryNorm(bounds, palette.N)
    layer = ax.scatter(cells.grid_x, cells.grid_y, c=cells.candidate_types,
                       cmap=palette, norm=norm, s=2.2, marker='s',
                       linewidths=0, rasterized=True)
    ax.set_title('b  Candidate types detected per cell', loc='left',
                 fontsize=11, fontweight='bold', color='#263432')
    bar = fig.colorbar(layer, ax=ax, fraction=.035, pad=.02,
                       ticks=[.5, 2, 4.5, 8.5, 15.5, (21 + upper) / 2])
    bar.ax.set_yticklabels(['0', '1-2', '3-5', '6-10', '11-20', '21+'])
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.05, .945, 'Dhaka local landscape-combination candidates',
             fontsize=17, fontweight='bold', color='#24312f')
    fig.text(.05, .900,
             f'{selected_count} Dhaka-only confirmed combinations | '
             f'{len(cells):,} observed 1-km cells | '
             f'{matching_panoramas:,} sampled panoramas match at least one',
             fontsize=10, color='#52615f')
    fig.text(.05, .064,
             'All sampled study-region cells are shown. A cell is colored if any sampled panorama matches. '
             'Patterns overlap and include nested pairs, triples, and quadruples.',
             fontsize=8.6, color='#586460')
    fig.text(.05, .042,
             'Only Dhaka passes the within-city replication cutoff for these candidates; '
             'below-cutoff responses in other cities remain possible.',
             fontsize=8.6, color='#586460')
    fig.savefig(output / 'Dhaka_Local_Overview.png', dpi=230, facecolor='white')
    fig.savefig(output / 'Dhaka_Local_Overview.pdf', facecolor='white')
    plt.close(fig)


def make_contact_sheets(records: list[dict], output: Path):
    font = ImageFont.truetype(PIL_FONT, 18)
    small = ImageFont.truetype(PIL_FONT, 14)
    pages = []
    for offset in range(0, len(records), 12):
        page_count = min(12, len(records) - offset)
        canvas = Image.new('RGB', (1816, 20 + ((page_count + 2) // 3) * 232),
                           '#ffffff')
        draw = ImageDraw.Draw(canvas)
        for slot, record in enumerate(records[offset:offset + 12]):
            x = 16 + (slot % 3) * 600
            y = 20 + (slot // 3) * 232
            color = ORDER_COLORS[record['order']]
            draw.rectangle((x, y, x + 575, y + 207), outline='#cfd7d4', width=1)
            draw.rectangle((x, y, x + 5, y + 207), fill=color)
            title = f"{record['index']:02d}  {record['english']}"
            if draw.textlength(title, font=font) > 540:
                title = title[:51] + '...'
            draw.text((x + 14, y + 10), title, fill='#273633', font=font)
            draw.text((x + 14, y + 35),
                      f"{record['cells']:,} cells | {record['panoramas']:,} panoramas | "
                      f"{record['city_rate']:.2%}", fill='#586561', font=small)
            original = Path(record['original'])
            with Image.open(original) as opened:
                thumb = opened.convert('RGB').resize((546, 146), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + 14, y + 56))
        page = offset // 12 + 1
        target = output / f'Dhaka_Local_Examples_P{page:02d}.png'
        canvas.save(target)
        pages.append(target.name)
    return pages


def run(stage: Path, frame_path: Path, results: Path, output: Path,
        device: str, limit: int | None):
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for four-view SAE attribution')
    (local, frame, assignment, scores, units, thresholds, city_positions,
     city_frame, cell_index, cells) = load_inputs(stage, frame_path, results)
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        local = local.iloc[:limit].copy()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'patterns').mkdir(exist_ok=True)
    n_cells = len(cells)
    richness = np.zeros(n_cells, dtype=np.int16)
    any_panorama = np.zeros(len(city_frame), dtype=bool)
    used_panoids = set()
    plans, long_counts = [], []
    for index, row in enumerate(local.itertuples(), 1):
        ids = json.loads(row.unit_ids)
        positive = pattern_mask(assignment, units, ids)[city_positions]
        counts = np.bincount(cell_index[positive], minlength=n_cells)
        if int(positive.sum()) != row.panoramas or int((counts > 0).sum()) != row.cells:
            raise ValueError(f'Saved Dhaka counts changed: {row.pattern_id}')
        richness += counts > 0
        any_panorama |= positive
        active = np.flatnonzero(counts)
        long_counts.append(pd.DataFrame({
            'pattern_id': row.pattern_id,
            'grid_x': cells.grid_x.to_numpy()[active],
            'grid_y': cells.grid_y.to_numpy()[active],
            'matching_panoramas': counts[active],
            'sampled_panoramas': cells.sampled_panoramas.to_numpy()[active]}))
        minimum_rank = 3 if row.pattern_id == 'cd8d335c382b' else 1
        sample, rank, strength = select_example(
            frame, scores, thresholds, city_positions, positive, ids,
            used_panoids, minimum_rank)
        plans.append({'index': index, 'pattern_id': row.pattern_id,
                      'label': row.label, 'english': english_label(row.label),
                      'unit_ids': ids, 'unit_labels': [units[uid]['label'] for uid in ids],
                      'order': int(row.order), 'panoramas': int(row.panoramas),
                      'cells': int(row.cells), 'city_rate': float(row.city_rate),
                      'sample': sample, 'strength_rank': rank,
                      'minimum_unit_score_ratio': strength,
                      'counts': counts})
    cells['candidate_types'] = richness
    cells['panoramas_with_any'] = np.bincount(
        cell_index[any_panorama], minlength=n_cells)
    cells.to_csv(output / 'Dhaka_Local_Grid_Summary.csv', index=False)
    pd.concat(long_counts, ignore_index=True).to_csv(
        output / 'Dhaka_Local_Grid_Pattern_Counts.csv', index=False)
    torch.set_num_threads(4)
    dino, sae, _ = load_models(device)
    records = []
    for plan in plans:
        sample = plan['sample']
        folder = output / 'patterns' / plan['pattern_id']
        original = folder / 'original.webp'
        overlay = folder / 'overlay.webp'
        metadata = folder / 'example.json'
        cached = False
        if metadata.is_file() and original.is_file() and overlay.is_file():
            saved = json.loads(metadata.read_text())
            cached = (saved.get('panoid') == str(sample.panoid) and
                      saved.get('checkpoint_sha256') == EXPECTED_CHECKPOINT and
                      saved.get('unit_ids') == plan['unit_ids'])
        if not cached:
            images = []
            for path in image_paths(CITY, sample.panoid):
                with Image.open(path) as opened:
                    images.append(opened.convert('RGB'))
            maps = infer(images, dino, sae, plan['unit_ids'], units, device)
            if maps.shape != (plan['order'], 4, 28, 28) or not np.isfinite(maps).all():
                raise ValueError(f'Invalid response map: {plan["pattern_id"]}')
            render(images, maps, folder)
            write_json(metadata, {'pattern_id': plan['pattern_id'],
                                  'panoid': str(sample.panoid),
                                  'unit_ids': plan['unit_ids'],
                                  'checkpoint_sha256': EXPECTED_CHECKPOINT})
        record = {key: value for key, value in plan.items()
                  if key not in {'sample', 'counts'}}
        record.update({'panoid': str(sample.panoid), 'grid_x': int(sample.grid_x),
                       'grid_y': int(sample.grid_y),
                       'original': str(original.resolve()),
                       'overlay': str(overlay.resolve()),
                       'figure': str((folder / 'map_and_example.png').resolve())})
        records.append(record)
        write_json(output / 'progress.json', {'processed': len(records),
                                              'total': len(plans)})
        print(f'{len(records)}/{len(plans)} {plan["pattern_id"]}', flush=True)
    atlas_path = output / 'Dhaka_Local_Combination_Atlas.pdf'
    with PdfPages(atlas_path) as pdf:
        for plan, record in zip(plans, records):
            folder = output / 'patterns' / plan['pattern_id']
            draw_page({**record, 'sample': plan['sample']}, cells,
                      plan['counts'], folder, folder / 'map_and_example.png', pdf)
    draw_overview(cells, output, len(plans), int(any_panorama.sum()))
    pages = make_contact_sheets(records, output)
    write_json(output / 'manifest.json', {
        'status': 'complete' if limit is None else 'limited_preview',
        'city': CITY, 'sampled_panoramas': len(city_frame),
        'observed_cells': n_cells, 'candidate_count': len(plans),
        'candidate_definition': 'Replicated within Dhaka and no other study city '
                                'passing the same within-city cutoff; not exclusive presence.',
        'grid_definition': 'Observed 1-km cells; counts are matching sampled panoramas '
                           'and candidate types can overlap within a cell.',
        'example_selection': 'Highest minimum normalized unit response with four images '
                             'and distinct panoid; cd8d335c382b starts at rank 3 after visual review.',
        'checkpoint_sha256': EXPECTED_CHECKPOINT,
        'panoramas_with_any_candidate': int(any_panorama.sum()),
        'cells_with_any_candidate': int((richness > 0).sum()),
        'contact_sheets': pages, 'patterns': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--frame', type=Path, default=FRAME)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.stage, args.frame, args.results, args.output, args.device, args.limit)
