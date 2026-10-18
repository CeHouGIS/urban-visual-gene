"""Mine recurrent relative-response compositions, independently of excess tests.

The observations are frozen-SAE panorama means, not object detections. Thresholds
are fitted once on discovery cities, with equal city/cell weights. Reusing the
earlier evaluation cities makes this entire revision exploratory.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .candidate_combinations import ELEMENTS, HIERARCHY, family_dimensions, stable_city
from .extract import DEFAULT_OUT, write_json
from .followup import FOLLOWUP, sha256

VERSION = 2
SCORE_VERSION = 1  # Raw response columns do not depend on downstream eligibility.
# Research scope: stable landscape fabric and fixed streetscape infrastructure.
# This is a reversible analysis filter, not a deletion of SAE dimensions.
EXCLUDED_FAMILIES = {'person': '活动人物不作为景观构成', 'vehicle': '车辆及其部件不作为景观构成'}
EXCLUDED_LABELS = {
 'construction': {'压路机', '工业设备', '拖拉机', '挖掘机', '施工机械', '水泥搅拌机',
                  '脚手架', '起重机', '集装箱'},
 'furniture': {'桌子', '移动厕所', '遮阳伞'},
 'sign': {'交通锥', '旗帜', '横幅'},
}


def landscape_exclusion(family, label):
    if family in EXCLUDED_FAMILIES:
        return EXCLUDED_FAMILIES[family]
    if label in EXCLUDED_LABELS.get(family, set()):
        return '移动、临时或未能确认固定性的对象不作为本轮景观构成'
    return None


def landscape_children(units, family):
    return [u for u in units if not u['fallback'] and u['family'] == family
            and u.get('landscape_eligible', True)]


def score_identity(run_identity, hierarchy_hash, units):
    # Eligibility changes mining, not the underlying all-4096 mass shares.
    raw_units = [{k: v for k, v in u.items() if k not in ('landscape_eligible', 'exclusion_reason')}
                 for u in units]
    return hashlib.sha256(json.dumps({'version': SCORE_VERSION, 'run': run_identity,
        'hierarchy': hierarchy_hash, 'units': raw_units}, sort_keys=True).encode()).hexdigest()
# Synonyms and object parts are merged before combinations are mined. Unlisted
# labels retain their original wording; this is a declared ontology, not VLM output.
ALIASES = {
 'building': {
  '建筑': '建筑物 房屋 建筑群 建筑结构 建筑立面 建筑墙面 建筑墙体 建筑外墙 建筑表面 建筑入口 建筑屋檐 建筑屋顶 屋顶 屋檐 屋顶结构 门 木门 铁门 卷帘门 门框 门廊 拱门 建筑拱门 窗户 玻璃窗 百叶窗 窗帘 阳台 阳台栏杆 建筑立柱 柱廊 空调外机 卫星天线 太阳能电池板 烟囱 水塔 遮阳篷',
  '高层建筑': '摩天大楼 高楼 高层住宅建筑群',
  '传统建筑': '传统中式建筑屋顶 传统建筑屋顶 古典建筑',
  '瓦顶建筑': '瓦片屋顶 红色屋顶',
  '金属屋顶建筑': '波纹金属屋顶',
 },
 'tree': {
  '乔木': '树 树木 树冠 树干 树叶 树皮 树皮纹理 树林 森林 林地 行道树',
  '棕榈类乔木': '棕榈树 棕榈树林 椰子树',
  '香蕉树': '香蕉叶', '针叶树': '松树 柏树',
 },
 'low_vegetation': {'草地': '草 草坪 草丛 地面植被', '灌木': '灌木丛',
                    '花卉': '花丛 花朵 橙色花朵 红色花朵 白色花朵'},
 'road': {'道路': '路面 道路标线 道路边缘'},
 'pedestrian': {'台阶与楼梯': '台阶 楼梯', '路缘石': '混凝土路缘石'},
 'barrier': {'围栏': '栅栏 栏杆 铁栅栏 金属围栏 金属栏杆',
             '护栏': '交通护栏 金属护栏', '铁丝网围栏': '铁丝网',
             '木栅栏': '白色栅栏'},
 'vehicle': {'车辆': '车 汽车 车轮及轮胎 轮胎 车窗 车门 车顶 车头 车头灯 车尾 车尾灯 车尾箱 尾灯 车灯 车身 车身侧面 车体 车体表面 车辆后部 车辆尾部 驾驶室',
             '卡车与货车': '卡车 卡车拖车 卡车车头 卡车货箱 卡车驾驶室 货车 白色货车 白色面包车 货车车厢 平板拖车 拖车'},
 'person': {'行人': '人 人物 人群 人脸 手 皮肤纹理 雨天行人'},
 'water': {'水体': '水面'},
 'terrain': {'山地': '山 山丘 山体 山坡 山脉', '岩石': '岩石表面 石块', '沙地': '沙土'},
 'transport_structure': {'桥梁': '桥 桥墩 桥面 桥梁结构 石拱桥',
                         '铁路': '铁路轨道 轨道 铁轨', '隧道': '隧道入口 隧道内壁'},
 'utility': {'输电塔': '铁塔 高压电线塔'},
 'sign': {'交通标志': '路牌 路标 停车标志 禁止停车 禁止驶入 三角形警示标志 高速公路指示牌',
          '商业招牌': '招牌 街道招牌 商店招牌'},
 'furniture': {'垃圾桶': '垃圾箱', '座椅': '长椅 椅子'},
 'construction': {'脚手架': '建筑脚手架', '起重机': '建筑起重机 塔式起重机',
                  '集装箱': '蓝色集装箱'},
}


def canonical(family, label):
    for target, aliases in ALIASES.get(family, {}).items():
        if label == target or label in aliases.split():
            return target
    return label


def make_units(hierarchy):
    eligible = {int(i) for v in family_dimensions(hierarchy).values() for i in v}
    grouped, originals = defaultdict(list), defaultdict(set)
    for row in hierarchy['dimensions']:
        if int(row['feature']) not in eligible:
            continue
        key = row['family'], canonical(row['family'], row['fine_label_zh'])
        grouped[key].append(int(row['feature']))
        originals[key].add(row['fine_label_zh'])
    units = []
    for (family, label), dims in sorted(grouped.items()):
        reason = landscape_exclusion(family, label)
        units.append({'id': len(units), 'key': family + ':' + label,
                      'family': family, 'label': label, 'dimensions': sorted(dims),
                      'source_labels': sorted(originals[family, label]), 'fallback': False,
                      'landscape_eligible': reason is None, 'exclusion_reason': reason})
    # A parent can be strongly responsive without a sufficiently responsive child.
    for family in ELEMENTS:
        allowed = bool(landscape_children(units, family))
        units.append({'id': len(units), 'key': family + ':*', 'family': family,
                      'label': next(r['label_zh'] for r in hierarchy['families'] if r['id'] == family) + '（综合响应）',
                      'dimensions': [], 'source_labels': [], 'fallback': True,
                      'landscape_eligible': allowed,
                      'exclusion_reason': None if allowed else '没有本轮允许的景观子类'})
    return units


def weighted_quantile(values, weights, q):
    positive = np.isfinite(values) & (values > 0) & (weights > 0)
    if not positive.any():
        return 1.  # a normalized share cannot exceed one; zero evidence stays off
    x, w = values[positive], weights[positive]
    order = np.argsort(x, kind='stable')
    x, w = x[order], w[order]
    return float(x[min(np.searchsorted(np.cumsum(w), q * w.sum()), len(x) - 1)])


def sampling_design(frame):
    keys = pd.MultiIndex.from_frame(frame[['city', 'grid_x', 'grid_y']])
    cell_id, cell_keys = pd.factorize(keys, sort=True)
    city_id, cities = pd.factorize(frame.city, sort=True)
    cell_n = np.bincount(cell_id)
    cells = pd.DataFrame(list(cell_keys), columns=['city', 'grid_x', 'grid_y'])
    cells['sampled_panoramas'] = cell_n
    city_cells = cells.groupby('city').size().to_dict()
    weights = np.asarray([1 / (city_cells[c] * cell_n[k]) for c, k in zip(frame.city, cell_id)])
    discovery = np.asarray([stable_city(c) % 3 != 0 for c in frame.city])
    return cell_id, city_id, list(cities), cells, weights, discovery


def calibrate(scores, units, weights, discovery, quantile=.6, floor=.001, parent_floor=.005):
    train_weights = weights * discovery
    thresholds = np.asarray([max(floor, weighted_quantile(scores[:, u['id']], train_weights, quantile))
                             if u.get('landscape_eligible', True) else 1.
                             for u in units if not u['fallback']])
    parents = np.column_stack([scores[:, [u['id'] for u in landscape_children(units, f)]].sum(axis=1)
                               for f in ELEMENTS])
    parent_thresholds = np.asarray([max(parent_floor, weighted_quantile(parents[:, i], train_weights, quantile))
                                    for i in range(len(ELEMENTS))])
    return thresholds, parent_thresholds, parents


def assign_units(scores, units, thresholds, parent_thresholds, parents):
    """One child per parent: same-object parts cannot form a composition."""
    assignment = np.full((len(scores), len(ELEMENTS)), -1, dtype=np.int16)
    for j, family in enumerate(ELEMENTS):
        ids = np.asarray([u['id'] for u in landscape_children(units, family)], dtype=int)
        if not len(ids):
            continue
        ratios = scores[:, ids] / thresholds[ids]
        best = ratios.argmax(axis=1)
        chosen = ids[best]
        fallback = next(u['id'] for u in units if u['fallback'] and u['family'] == family)
        chosen = np.where(ratios[np.arange(len(scores)), best] >= 1, chosen, fallback)
        assignment[parents[:, j] >= parent_thresholds[j], j] = chosen[parents[:, j] >= parent_thresholds[j]]
    return assignment


def packed_mask(mask):
    return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')


def unpack_mask(bits, n):
    return np.unpackbits(np.frombuffer(bits.to_bytes((n + 7) // 8, 'little'), dtype=np.uint8),
                         bitorder='little', count=n).astype(bool)


def pattern_mask(assignment, units, pattern):
    mask = np.ones(len(assignment), dtype=bool)
    for unit in pattern:
        mask &= assignment[:, ELEMENTS.index(units[unit]['family'])] == unit
    return mask


def mine(assignment, units, weights, discovery, city_id, cell_id, min_rate=.005, max_order=4):
    n = len(assignment)
    w_disc = weights * discovery / len(np.unique(city_id[discovery]))
    bits = {u['id']: packed_mask((assignment[:, ELEMENTS.index(u['family'])] == u['id']) & discovery)
            for u in units if u.get('landscape_eligible', True)}
    level = {(i,): mask for i, mask in bits.items() if mask.bit_count() >= 30
             and w_disc[unpack_mask(mask, n)].sum() >= min_rate}
    results = []
    for order in range(2, max_order + 1):
        keys = sorted(level)
        next_level = {}
        groups = defaultdict(list)
        for key in keys:
            groups[key[:-1]].append(key)
        for members in groups.values():
            for first, second in itertools.combinations(members, 2):
                candidate = first + (second[-1],)
                if len({units[i]['family'] for i in candidate}) != order:
                    continue
                if any(candidate[:i] + candidate[i + 1:] not in level for i in range(order)):
                    continue
                mask = level[first] & level[second]
                if mask.bit_count() < 30:
                    continue
                positive = unpack_mask(mask, n)
                rate = float(w_disc[positive].sum())
                if rate < min_rate or len(np.unique(city_id[positive])) < 3 or len(np.unique(cell_id[positive])) < 8:
                    continue
                next_level[candidate] = mask
                results.append({'units': list(candidate), 'order': order, 'discovery_rate': rate,
                                'discovery_panoramas': mask.bit_count()})
        print(f'Discovery mining: order {order}; {len(next_level)} recurrent combinations', flush=True)
        level = next_level
    return results


def load_scores(args, units, source_frame):
    identity = score_identity(args.run_identity, sha256(args.hierarchy), units)
    cache = args.output / 'scores.npy'
    meta = args.output / 'scores_identity.json'
    n_units = sum(not u['fallback'] for u in units)
    if meta.exists():
        value = json.loads(meta.read_text())
        if value != {'identity': identity, 'rows': len(source_frame), 'units': n_units}:
            raise ValueError('Score cache identity differs')
        return np.load(cache, mmap_mode='r'), identity
    values = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32,
                                     shape=(len(source_frame), n_units))
    cursor = 0
    run = json.loads((args.extraction / 'run.json').read_text())
    for entry in run['cities']:
        city = entry['city']
        expected = source_frame[source_frame.city == city]
        city_count = 0
        for path in sorted((args.extraction / city.replace('/', '__')).glob('*/vectors.npz')):
            marker = json.loads(path.with_name('complete.json').read_text())
            if marker['identity'] != args.run_identity or marker['archive_bytes'] != path.stat().st_size:
                raise ValueError('Mismatched source chunk')
            with np.load(path, allow_pickle=False) as data:
                n = len(data['panoid'])
                rows = source_frame.iloc[cursor:cursor+n]
                if not ((rows.city == city).all() and np.array_equal(rows.panoid, data['panoid'])
                        and np.array_equal(rows.grid_x, data['grid_x']) and np.array_equal(rows.grid_y, data['grid_y'])):
                    raise ValueError('Score and metadata order mismatch')
                x = data['composition'].astype(np.float32)
                total = x.sum(axis=1, dtype=np.float64)
                for unit in units[:n_units]:
                    mass = x[:, unit['dimensions']].sum(axis=1, dtype=np.float64)
                    values[cursor:cursor+n, unit['id']] = np.divide(mass, total, out=np.zeros(n), where=total>0)
                cursor += n
                city_count += n
        if city_count != len(expected):
            raise ValueError(f'Missing city rows: {city}')
        values.flush()
        print(f'Fine-label scores: {city}; {cursor:,}/{len(source_frame):,}', flush=True)
        write_json(args.output / 'progress.json', {'status': 'scoring', 'panoramas_done': cursor, 'total': len(source_frame)})
    if cursor != len(source_frame):
        raise ValueError('Incomplete corpus')
    write_json(meta, {'identity': identity, 'rows': len(source_frame), 'units': n_units})
    return values, identity


def summarise_pattern(mask, weights, city_id, cell_id, discovery):
    result = {}
    for split, chosen in [('discovery', discovery), ('evaluation', ~discovery)]:
        positive = mask & chosen
        result.update({f'{split}_rate': float(weights[positive].sum() / len(np.unique(city_id[chosen]))),
                       f'{split}_panoramas': int(positive.sum()),
                       f'{split}_cells': int(len(np.unique(cell_id[positive]))),
                       f'{split}_cities': int(len(np.unique(city_id[positive])))})
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'complete.json').unlink(missing_ok=True)
    (args.output / 'evidence_complete.json').unlink(missing_ok=True)
    source = args.followup / 'stages/candidate_combinations'
    protocol = json.loads((source / 'protocol.json').read_text())
    verified = json.loads((args.followup / 'complete.json').read_text())
    if protocol['run_identity'] != args.run_identity or verified['identity']['extract_run_identity'] != args.run_identity:
        raise ValueError('Mismatched corpus identity')
    frame = pd.read_parquet(source / 'panorama_elements.parquet')
    if frame.city.nunique() != 30 or len(frame) != verified['processed'] or frame.duplicated(['city', 'panoid']).any():
        raise ValueError('Incomplete or duplicated 30-city corpus')
    hierarchy = json.loads(args.hierarchy.read_text())
    units = make_units(hierarchy)
    write_json(args.output / 'units.json', units)
    scores, identity = load_scores(args, units, frame)
    cell_id, city_id, cities, cells, weights, discovery = sampling_design(frame)
    thresholds, parent_thresholds, parents = calibrate(scores, units, weights, discovery, args.quantile, args.floor, args.parent_floor)
    assignment = assign_units(scores, units, thresholds, parent_thresholds, parents)
    np.save(args.output / 'assignment.npy', assignment)
    calibration = {'quantile': args.quantile, 'absolute_mass_floor': args.floor, 'parent_mass_floor': args.parent_floor,
                   'unit_thresholds': thresholds.tolist(), 'family_thresholds': dict(zip(ELEMENTS, parent_thresholds.tolist())),
                   'fit_cities': sorted(frame.city[discovery].unique()),
                   'evaluation_cities': sorted(frame.city[~discovery].unique()), 'city_specific_thresholds': False}
    write_json(args.output / 'calibration.json', calibration)
    candidates = mine(assignment, units, weights, discovery, city_id, cell_id, args.min_rate, args.max_order)
    # Compute null means as descriptive enrichment only: no p-values or fresh
    # confirmatory claim are attached to this exploratory algorithm revision.
    block_keys = pd.MultiIndex.from_arrays([frame.city, frame.grid_x // 5, frame.grid_y // 5])
    block_id, _ = pd.factorize(block_keys, sort=True)
    n_blocks = block_id.max() + 1
    block_n = np.bincount(block_id)
    counts = np.column_stack([np.bincount(block_id, weights=(assignment[:, ELEMENTS.index(u['family'])] == u['id']), minlength=n_blocks)
                              for u in units])
    fractions = counts / block_n[:, None]
    rows = []
    for candidate in candidates:
        pattern = candidate['units']
        mask = pattern_mask(assignment, units, pattern)
        stats = summarise_pattern(mask, weights, city_id, cell_id, discovery)
        expected = float((block_n * fractions[:, pattern].prod(axis=1)).sum())
        qualifies = (stats['evaluation_rate'] >= args.min_rate and stats['evaluation_panoramas'] >= 30
                     and stats['evaluation_cells'] >= 8 and stats['evaluation_cities'] >= 3)
        key = '+'.join(units[i]['key'] for i in pattern)
        rows.append({'pattern_id': hashlib.sha256(key.encode()).hexdigest()[:12],
                     'label': ' ＋ '.join(units[i]['label'] for i in pattern), 'key': key,
                     'units': pattern, 'order': len(pattern), **stats,
                     'positive_panoramas': int(mask.sum()),
                     'positive_cells': int(len(np.unique(cell_id[mask]))),
                     'positive_cities': int(len(np.unique(city_id[mask]))),
                     'block_null_expected': expected,
                     'block_lift': float(mask.sum()/expected) if expected else 0.,
                     'recurrent': bool(qualifies), 'near_redundant_with': None,
                     'inference': 'exploratory_relative_response', 'layout_verified': False})
    # Near-closed sets: a subset is redundant if a discovered extension retains
    # at least 90% of its weighted support. Evaluation support does not set this rule.
    for row in rows:
        if row['order'] == 2 or row['order'] == 3:
            extensions = [r for r in rows if r['order'] > row['order'] and set(row['units']) < set(r['units'])
                          and r['discovery_rate'] >= .9 * row['discovery_rate']]
            if extensions:
                best = max(extensions, key=lambda r: (r['order'], r['discovery_rate']))
                row['near_redundant_with'] = best['pattern_id']
    retained = [r for r in rows if r['recurrent'] and not r['near_redundant_with']]
    retained.sort(key=lambda r: (-r['order'], -min(r['discovery_rate'], r['evaluation_rate']), r['pattern_id']))
    write_json(args.output / 'patterns.json', retained)
    pd.DataFrame([{**r, 'units': json.dumps(r['units'])} for r in rows]).to_csv(args.output / 'all_discovered_patterns.csv', index=False)
    city_rows, cell_values = [], []
    coverage = np.zeros(len(frame), dtype=bool)
    for row in retained:
        mask = pattern_mask(assignment, units, row['units'])
        coverage |= mask
        pos = np.bincount(cell_id, weights=mask, minlength=len(cells)).astype(int)
        rate = pos / cells.sampled_panoramas.to_numpy()
        # Sampling cap is ten, so storing positive counts in uint8 is lossless.
        cell_values.append({'pattern_id': row['pattern_id'], 'positive_counts_u8': base64.b64encode(pos.astype('uint8').tobytes()).decode()})
        for c, city in enumerate(cities):
            selected_cells = cells.city == city
            selected_panos = city_id == c
            city_rows.append({'pattern_id': row['pattern_id'], 'city': city,
                'cell_weighted_rate': float(rate[selected_cells].mean()),
                'positive_panoramas': int(mask[selected_panos].sum()), 'sampled_panoramas': int(selected_panos.sum()),
                'positive_cells': int((pos[selected_cells] > 0).sum()), 'observed_cells': int(selected_cells.sum())})
    pd.DataFrame(city_rows).to_csv(args.output / 'city_distribution.csv', index=False)
    cells.to_csv(args.output / 'observed_cells.csv', index=False)
    write_json(args.output / 'cell_patterns.json', cell_values)
    coverage_rows = []
    for c, city in enumerate(cities):
        selected = city_id == c
        cw = np.bincount(cell_id[selected], weights=coverage[selected], minlength=len(cells))
        cmask = cells.city == city
        coverage_rows.append({'city': city, 'panoramas': int(selected.sum()),
                              'covered_panoramas': int(coverage[selected].sum()),
                              'cell_weighted_coverage': float((cw[cmask]/cells.sampled_panoramas[cmask]).mean())})
    pd.DataFrame(coverage_rows).to_csv(args.output / 'coverage_by_city.csv', index=False)
    sensitivity = []
    for q in sorted({.5, args.quantile, .7}):
        if q == args.quantile:
            assigned = assignment
        else:
            tau, pt, par = calibrate(scores, units, weights, discovery, q, args.floor, args.parent_floor)
            assigned = assign_units(scores, units, tau, pt, par)
        union = np.zeros(len(frame), dtype=bool)
        recurring = 0
        for row in retained:
            mask = pattern_mask(assigned, units, row['units'])
            union |= mask
            stats = summarise_pattern(mask, weights, city_id, cell_id, discovery)
            recurring += stats['evaluation_rate'] >= args.min_rate and stats['evaluation_cities'] >= 3 and stats['evaluation_panoramas'] >= 30 and stats['evaluation_cells'] >= 8
        sensitivity.append({'quantile': q, 'frozen_patterns': len(retained), 'patterns_repeated_evaluation': int(recurring),
                            'equal_city_cell_coverage': float(weights[union].sum()/30),
                            'panorama_coverage': float(union.mean())})
    write_json(args.output / 'sensitivity.json', sensitivity)
    old = pd.read_csv(source / 'candidates.csv')
    old = old[old.status == 'co_presence_candidate']
    old_union = np.zeros(len(frame), dtype=bool)
    for r in old.itertuples():
        old_union |= frame[f'present_{r.element_a}'].to_numpy() & frame[f'present_{r.element_b}'].to_numpy()
    diagnostics = []
    for i, family in enumerate(ELEMENTS):
        diagnostics.append({'family': family, 'old_2pct_response_rate': float(frame[f'present_{family}'].mean()),
                            'new_response_rate': float((assignment[:, i]>=0).mean()),
                            'new_mass_threshold': float(parent_thresholds[i])})
    pd.DataFrame(diagnostics).to_csv(args.output / 'threshold_diagnostics.csv', index=False)
    result = {'version': VERSION, 'run_identity': args.run_identity, 'score_identity': identity,
              'cities': 30, 'panoramas': len(frame), 'cells': len(cells),
              'fine_source_labels': len({(r['family'], r['fine_label_zh']) for r in hierarchy['dimensions'] if r['provisional_graph_candidate']}),
              'raw_canonical_units': sum(not u['fallback'] for u in units),
              'canonical_units': sum(not u['fallback'] and u['landscape_eligible'] for u in units),
              'excluded_units': sum(not u['fallback'] and not u['landscape_eligible'] for u in units),
              'landscape_filter_applied': True,
              'discovered_patterns': len(rows), 'recurrent_patterns': len(retained),
              'orders': {str(k): sum(r['order']==k for r in retained) for k in range(2, args.max_order+1)},
              'panorama_coverage': float(coverage.mean()), 'equal_city_cell_coverage': float(weights[coverage].sum()/30),
              'old_seven_panorama_coverage': float(old_union.mean()),
              'old_seven_equal_city_cell_coverage': float(weights[old_union].sum()/30),
              'coverage_unit': 'panoramas matching at least one composition; NOT patch coverage or semantic object recall',
              'labels_human_validated': False, 'layout_verified': False, 'statistical_confirmation': False,
              'interpretation': 'Exploratory recurrent relative-response compositions, not object prevalence or validated prototypes.'}
    write_json(args.output / 'summary.json', result)
    write_json(args.output / 'protocol.json', {'version': VERSION, 'run_identity': args.run_identity,
        'hierarchy_sha256': sha256(args.hierarchy), 'calibration': calibration,
        'weighting': 'equal cities; equal observed 1-km cells per city; equal panoramas per cell',
        'ontology': ALIASES, 'composition': 'one winning normalized subtype per responsive parent; at most one unit per parent',
        'landscape_scope': 'Stable landscape fabric and fixed streetscape infrastructure; no people, vehicles, containers, construction machinery or listed temporary objects.',
        'excluded_families': EXCLUDED_FAMILIES,
        'excluded_labels': {k: sorted(v) for k, v in EXCLUDED_LABELS.items()},
        'excluded_units': [{k: u[k] for k in ('id', 'key', 'dimensions', 'exclusion_reason')}
                           for u in units if not u['fallback'] and not u['landscape_eligible']],
        'parent_fallback': 'Only eligible child dimensions contribute; excluded responses cannot re-enter through a parent fallback. All-4096 denominator is unchanged.',
        'winning_score': 'child all-4096 mass share divided by its discovery-city threshold; winner must exceed one',
        'search': f'Apriori orders 2..{args.max_order}; weighted support >= {args.min_rate} in discovery; same evaluation support rule',
        'support_minima': {'panoramas': 30, 'cells': 8, 'cities': 3}, 'near_closed_tolerance': .9,
        'significance': 'Not a significance filter. Block-independent marginal null mean and lift are descriptive only.',
        'evaluation_reused': True, 'status': 'exploratory_algorithm_revision',
        'limitations': ['Relative response is not an object detection or a presence threshold.',
          'Quantile calibration does not establish false-positive or false-negative rates.',
          'Original Qwen labels remain provisional; parts are merged but semantic errors can persist.',
          'Four directions are pooled, so object counts and layout/roadside relations are not established.',
          'Evaluation cities have been examined before; fresh spatial or external validation remains necessary.',
          'Coverage is panorama retrieval coverage, not the 80% patch explanation target.']})
    write_json(args.output / 'complete.json', {'stage': 'composition_patterns', 'status': 'complete',
        'run_identity': args.run_identity, 'evidence': result})
    write_json(args.output / 'progress.json', {'status': 'complete', **result})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    p.add_argument('--followup', type=Path, default=FOLLOWUP)
    p.add_argument('--hierarchy', type=Path, default=HIERARCHY)
    p.add_argument('--output', type=Path, default=FOLLOWUP/'stages/composition_patterns')
    p.add_argument('--run-identity', required=True)
    p.add_argument('--quantile', type=float, default=.6)
    p.add_argument('--floor', type=float, default=.001, help='Minimum child mass share (not semantic presence)')
    p.add_argument('--parent-floor', type=float, default=.005, help='Minimum parent mass share; suppress weak diffuse responses')
    p.add_argument('--min-rate', type=float, default=.005)
    p.add_argument('--max-order', type=int, default=4)
    args = p.parse_args()
    if not 0 < args.quantile < 1 or not 0 < args.floor <= args.parent_floor < 1 or not 0 < args.min_rate < 1 or not 2 <= args.max_order <= 4:
        p.error('Invalid calibration/support/order settings')
    run(args)
