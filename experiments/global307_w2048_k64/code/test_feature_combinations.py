"""Held-out city test of co-occurring W2048/k64 street-view SAE features.

Encode each complete four-view panorama with its own BatchTopK budget. Discover
candidate feature pairs in one set of cities, then test frozen candidates in
other cities. The conditional permutation null shuffles panorama assignments
within city and activation-richness strata; it keeps each feature's prevalence
in every stratum while approximately controlling overall feature richness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from formal.analysis.plot_feature_activation_maps import load_checkpoint


CHECKPOINT = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_w2048_k64/sae_448_k2048.pt')
ANCHORS = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_scaling_heldout_295city_40pano/anchor_manifest.json')
FEATURES = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/formal_batchtopk_307city_balanced_3000')
OUT = Path('reports/data/feature_combinations_w2048_k64')
SEMANTIC = Path('reports/data/semantic_umap_w2048_k64_top100/feature_semantic_embeddings.f32.npy')
ROWS_PER_PANO = 3136


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def panorama_records():
    rows = json.loads(ANCHORS.read_text())
    result = []
    seen = set()
    for row in rows:
        key = (row['city'], int(row['source_pano']))
        if key in seen:
            continue
        seen.add(key)
        result.append({'city': key[0], 'source_pano': key[1], 'filtered_pano': int(row['filtered_pano'])})
    del rows
    cities = defaultdict(int)
    for row in result:
        cities[row['city']] += 1
    if len(result) != 11800 or len(cities) != 295 or set(cities.values()) != {40}:
        raise ValueError(f'Unexpected held-out cohort: {len(result)} panoramas, {len(cities)} cities, counts {set(cities.values())}')
    return result


def encode(args, records):
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device).half()
    if (model.input_dim, model.latent_dim, model.k) != (1024, 2048, 64):
        raise ValueError('Expected the W2048/k64 checkpoint with D1024')
    by_city = defaultdict(list)
    for row in records:
        by_city[row['city']].append(row)
    for city_index, (city, city_rows) in enumerate(sorted(by_city.items()), 1):
        output = args.out / 'city_counts' / f'{city}.npy'
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            if np.load(output, mmap_mode='r').shape != (40, 2048):
                raise ValueError(f'Invalid cached count shape: {output}')
            continue
        source_path = FEATURES / f'dict_{city}_p784.f16.npy'
        source = np.load(source_path, mmap_mode='r')
        counts = np.empty((40, 2048), dtype=np.uint16)
        with torch.inference_mode():
            for i, row in enumerate(city_rows):
                start = row['source_pano'] * ROWS_PER_PANO
                block = np.asarray(source[start:start + ROWS_PER_PANO])
                if block.shape != (ROWS_PER_PANO, 1024):
                    raise ValueError(f'Incomplete feature block: {city} {row}')
                x = torch.from_numpy(block.copy()).to(device)
                positive = torch.relu(model._preactivate(x)).reshape(-1)
                values, indices = torch.topk(positive, model.k * ROWS_PER_PANO, sorted=False)
                active = indices[values > 0].remainder(model.latent_dim)
                counts[i] = torch.bincount(active, minlength=model.latent_dim).cpu().numpy().astype(np.uint16)
                if (i + 1) % 10 == 0:
                    print(f'encode {city_index}/295 {city} {i+1}/40', flush=True)
        temporary = output.with_suffix('.tmp.npy')
        np.save(temporary, counts)
        temporary.replace(output)
    (args.out / 'records.json').write_text(json.dumps(records, ensure_ascii=False))
    (args.out / 'provenance.json').write_text(json.dumps({
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'anchor_manifest': str(ANCHORS), 'anchor_manifest_sha256': sha256(ANCHORS),
        'unit': 'complete four-view panorama', 'patches_per_panorama': ROWS_PER_PANO,
        'activation': 'positive patches selected by panorama-level BatchTopK k64',
    }, indent=2))


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    return np.minimum(q, 1)


def stratification(cities, richness):
    groups = []
    for city in sorted(set(cities)):
        ix = np.flatnonzero(cities == city)
        order = ix[np.argsort(richness[ix], kind='stable')]
        groups.extend(g for g in np.array_split(order, 4) if len(g) > 1)
    return groups


def observed_and_expected(x, y, groups):
    observed = np.zeros(x.shape[1], dtype=np.float64)
    expected = np.zeros(x.shape[1], dtype=np.float64)
    for ix in groups:
        observed += np.sum(x[ix] & y[ix], axis=0)
        expected += x[ix].sum(axis=0) * y[ix].sum(axis=0) / len(ix)
    return observed, expected


def test(args, records):
    cities = np.array([r['city'] for r in records])
    counts = np.concatenate([np.load(args.out / 'city_counts' / f'{c}.npy') for c in sorted(set(cities))])
    if counts.shape != (len(records), 2048) or list(cities) != sorted(cities):
        raise ValueError('Count archive and record order disagree')
    rng = np.random.default_rng(args.seed)
    city_names = np.array(sorted(set(cities)))
    rng.shuffle(city_names)
    discovery_cities = set(city_names[:207])
    discovery = np.isin(cities, list(discovery_cities))
    validation = ~discovery
    # Frozen per-feature count cutoffs are based solely on discovery cities.
    thresholds = np.quantile(counts[discovery], .90, axis=0, method='higher')
    thresholds = np.maximum(thresholds, 1)
    binary = counts >= thresholds[None]
    freq = binary[discovery].mean(axis=0)
    eligible = (freq >= .025) & (freq <= .20)
    ids = np.flatnonzero(eligible)
    d = binary[discovery][:, ids].astype(np.uint8)
    n = len(d)
    overlap = d.T.astype(np.int32) @ d.astype(np.int32)
    prevalence = d.sum(axis=0)
    expected = prevalence[:, None] * prevalence[None] / n
    excess = overlap - expected
    np.fill_diagonal(excess, -np.inf)
    left, right = np.triu_indices(len(ids), 1)
    semantic = np.load(SEMANTIC)
    if semantic.shape[0] != 2048:
        raise ValueError('Semantic embedding does not match SAE dictionary')
    semantic = semantic / np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-9)
    semantic_cosine = np.sum(semantic[ids[left]] * semantic[ids[right]], axis=1)
    keep = (overlap[left, right] >= 30) & (excess[left, right] >= 15) & (semantic_cosine < args.max_semantic_cosine)
    order = np.argsort(excess[left[keep], right[keep]])[::-1][:args.candidates]
    selected_cosine = semantic_cosine[keep][order]
    left, right = left[keep][order], right[keep][order]
    if not len(left):
        raise ValueError('No discovery candidates at the fixed support thresholds')
    a, b = ids[left], ids[right]
    v = binary[validation]
    vcity = cities[validation]
    richness = v.sum(axis=1)
    groups = stratification(vcity, richness)
    x, y = v[:, a], v[:, b]
    obs, exp = observed_and_expected(x, y, groups)
    null_ge = np.zeros(len(a), dtype=np.int32)
    perm = np.arange(len(v))
    for iteration in range(args.permutations):
        for ix in groups:
            perm[ix] = rng.permutation(ix)
        shuffled = np.sum(x & y[perm], axis=0)
        null_ge += shuffled >= obs
        if (iteration + 1) % 1000 == 0:
            print(f'validation permutations {iteration+1}/{args.permutations}', flush=True)
    p = (null_ge + 1) / (args.permutations + 1)
    q = bh(p)
    rows = []
    for z, (i, j) in enumerate(zip(a, b)):
        supports = []
        for city in sorted(set(vcity)):
            ix = vcity == city
            supports.append(int(np.sum(v[ix, i] & v[ix, j])))
        rows.append({
            'feature_a': int(i), 'feature_b': int(j),
            'semantic_cosine': float(selected_cosine[z]),
            'discovery_overlap': int(overlap[left[z], right[z]]),
            'discovery_excess': round(float(excess[left[z], right[z]]), 3),
            'validation_overlap': int(obs[z]), 'validation_expected': round(float(exp[z]), 3),
            'validation_excess': round(float(obs[z] - exp[z]), 3),
            'validation_lift': round(float(obs[z] / max(exp[z], 1e-12)), 3),
            'validation_city_support': int(sum(s > 0 for s in supports)),
            'permutation_p': float(p[z]), 'fdr_q': float(q[z]),
        })
    rows.sort(key=lambda r: (r['fdr_q'], -r['validation_excess']))
    with (args.out / 'pair_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.out / 'analysis_arrays.npz', thresholds=thresholds, eligible=eligible,
                        discovery_city=np.array(sorted(discovery_cities)), validation_city=np.array(sorted(set(vcity))))
    summary = {
        'model': 'W2048/k64', 'heldout_panoramas': len(records), 'cities': len(city_names),
        'semantic_embedding': str(SEMANTIC), 'semantic_embedding_sha256': sha256(SEMANTIC),
        'seed': args.seed,
        'discovery_cities': len(discovery_cities), 'validation_cities': len(set(vcity)),
        'eligible_features': len(ids), 'candidates': len(rows),
        'max_semantic_cosine': args.max_semantic_cosine,
        'permutations': args.permutations, 'fdr_05_pairs': sum(r['fdr_q'] < .05 for r in rows),
        'threshold': 'feature-specific 90th percentile of patch counts on discovery panoramas, minimum 1',
        'null': 'permute one feature within city and panorama activation-richness quartile',
        'limitation': 'richness matched in quartiles, not exact row-degree preserving; co-occurrence alone does not establish spatial template or causal relation',
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['encode', 'test', 'all'], default='all')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--candidates', type=int, default=100)
    parser.add_argument('--max-semantic-cosine', type=float, default=.85)
    parser.add_argument('--permutations', type=int, default=9999)
    args = parser.parse_args()
    records = panorama_records()
    if args.stage in ('encode', 'all'):
        encode(args, records)
    if args.stage in ('test', 'all'):
        test(args, records)


if __name__ == '__main__':
    main()
