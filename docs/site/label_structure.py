"""Separate Qwen label taxonomy, part-whole relations, and peer groups."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RULES = Path(__file__).with_name('label_structure_rules.json')
TEMPLATES = Path(__file__).with_name('templates')


def build_structure(graph, rules):
    if rules['version'] != 1:
        raise ValueError('Unsupported label structure rules')
    families = {family['id']: family for family in graph['families']}
    labels = {(family['id'], label['label']): label
              for family in graph['families'] for label in family['labels'] if label['eligible']}
    hierarchy, parallel, linked, seen_edges = [], [], set(), set()

    def entry(family, label):
        row = labels.get((family, label))
        return {'label': label, 'count': row['count'], 'features': row['features']} if row else None

    for rule in rules['hierarchy']:
        family, parent, relation = rule['family'], rule['parent'], rule['relation']
        if family not in families or relation not in {'is_a', 'part_of'}:
            raise ValueError(f'Invalid hierarchy rule: {rule}')
        if len(rule['children']) != len(set(rule['children'])) or parent in rule['children']:
            raise ValueError(f'Duplicate or self-referential children: {rule}')
        if not entry(family, parent):
            continue
        children = []
        for child in rule['children']:
            item = entry(family, child)
            if not item:
                continue
            edge = (family, parent, child, relation)
            if edge in seen_edges:
                raise ValueError(f'Duplicate label relation: {edge}')
            seen_edges.add(edge)
            children.append(item)
            linked.update(((family, parent), (family, child)))
        if children:
            hierarchy.append({'family': family, 'family_label': families[family]['label'],
                              'parent': entry(family, parent), 'relation': relation,
                              'children': sorted(children, key=lambda row: (-row['count'], row['label']))})
            if len(children) >= 2:
                parallel.append({'family': family, 'family_label': families[family]['label'],
                                 'name': parent + ('的类型' if relation == 'is_a' else '的部件'),
                                 'basis': 'shared_parent', 'parent': parent,
                                 'relation': relation, 'labels': children})

    for rule in rules['parallel']:
        family = rule['family']
        if family not in families or len(rule['labels']) != len(set(rule['labels'])):
            raise ValueError(f'Invalid peer group: {rule}')
        members = [entry(family, label) for label in rule['labels']]
        members = [row for row in members if row]
        if len(members) < 2:
            continue
        linked.update((family, row['label']) for row in members)
        parallel.append({'family': family, 'family_label': families[family]['label'],
                         'name': rule['name'], 'basis': 'same_level_curated',
                         'parent': None, 'relation': None,
                         'labels': sorted(members, key=lambda row: (-row['count'], row['label']))})

    unlinked = [{'family': family, 'family_label': families[family]['label'], **entry(family, label)}
                for family, label in labels if (family, label) not in linked]
    unlinked.sort(key=lambda row: (-row['count'], row['family'], row['label']))
    hierarchy.sort(key=lambda row: (-row['parent']['count'], row['family'], row['parent']['label'], row['relation']))
    parallel.sort(key=lambda row: (-sum(label['count'] for label in row['labels']), row['family'], row['name']))
    return {'source_identity': graph['identity'],
            'rules_sha256': hashlib.sha256(json.dumps(rules, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'method': 'Explicit Qwen-label relations only; same-family membership alone is not a semantic edge',
            'validation': 'provisional_text_only',
            'counts': {'eligible_labels': len(labels), 'linked_labels': len(linked),
                       'unlinked_labels': len(unlinked), 'hierarchy_groups': len(hierarchy),
                       'hierarchy_edges': sum(len(row['children']) for row in hierarchy),
                       'parallel_groups': len(parallel)},
            'families': [{'id': family['id'], 'label': family['label']} for family in graph['families']],
            'hierarchy': hierarchy, 'parallel': parallel, 'unlinked': unlinked}


def install(site, render_page, snapshot):
    graph = json.loads((site / 'elements/category-graph.json').read_text())
    hierarchy = json.loads((site / 'elements/hierarchy.json').read_text())
    if graph['identity'] != hierarchy['identity'] or graph['dimension_screening_identity'] != hierarchy['dimension_screening_identity']:
        raise ValueError('Qwen label graph and dimension evidence do not match')
    rules = json.loads(RULES.read_text())
    result = build_structure(graph, rules)
    target = site / 'label-structure'
    target.mkdir(exist_ok=True)
    (target / 'structure.json').write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    with (target / 'hierarchy.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['family', 'parent', 'child', 'relation', 'child_dimension_count', 'child_features'])
        for group in result['hierarchy']:
            for child in group['children']:
                writer.writerow([group['family'], group['parent']['label'], child['label'], group['relation'],
                                 child['count'], ' '.join('f' + str(feature) for feature in child['features'])])
    with (target / 'parallel.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['family', 'peer_group', 'basis', 'label', 'dimension_count', 'features'])
        for group in result['parallel']:
            for label in group['labels']:
                writer.writerow([group['family'], group['name'], group['basis'], label['label'],
                                 label['count'], ' '.join('f' + str(feature) for feature in label['features'])])
    content = f'''<div class="eyebrow">Qwen label structure</div><h1>标签的并列与上下级关系</h1>
<p class="lead">区分同层标签、某类的具体类型，以及整体的组成部件。</p>
<p class="note">依据现有 Qwen 标签做保守的文字关系整理：{result['counts']['hierarchy_edges']} 条上下级关系、{result['counts']['parallel_groups']} 组并列关系。窗户是建筑的部件，不是建筑的一种；屋顶类型之间才是同层并列。父类只用于导航，不自动构成语义上下级。所有关系仍待图像证据和人工核查。</p>
<div class="structure-controls"><div class="structure-tabs" aria-label="关系视图"><button id="hierarchy-tab" type="button" aria-pressed="true">上下级</button><button id="parallel-tab" type="button" aria-pressed="false">同层并列</button><button id="unlinked-tab" type="button" aria-pressed="false">未判定</button></div><select id="structure-family" aria-label="筛选类别"><option value="">全部类别</option></select><input id="structure-search" aria-label="搜索标签" placeholder="搜索 Qwen 标签"></div>
<p id="structure-count" class="muted" role="status"></p><div class="structure-layout"><div id="structure-list" class="structure-list" aria-label="关系列表"></div><section id="structure-detail" class="structure-detail" aria-label="关系详情"></section></div>
<p class="muted">{result['counts']['unlinked_labels']} 个有资格展示的标签尚未指定可靠关系，保留在“未判定”。数字是 SAE 维度数，不是街景物体数量。<a href="hierarchy.csv">下载上下级 CSV</a> · <a href="parallel.csv">下载并列 CSV</a> · <a href="structure.json">完整数据与来源</a></p>'''
    document = render_page('标签结构', 'label-structure', content, snapshot,
                           '<script src="structure.js" defer></script>')
    document = document.replace('</head>', '<link rel="stylesheet" href="structure.css"></head>')
    (target / 'index.html').write_text(document)
    for suffix in ['css', 'js']:
        shutil.copyfile(TEMPLATES / f'label-structure.{suffix}', target / f'structure.{suffix}')
    return result['counts']
