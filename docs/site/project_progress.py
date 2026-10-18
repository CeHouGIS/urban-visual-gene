#!/usr/bin/env python3
"""Read bounded experiment status files and publish a small public progress snapshot.

No model imports, image scans, CUDA contexts, or training mutations are needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).with_name('templates')
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'
RUNS = [('real_seed42', '真实数据 · seed 42'), ('real_seed43', '真实数据 · seed 43'),
        ('spatial_shuffle_seed42', '空间打乱对照'), ('feature_shuffle_seed42', '维度对应打乱对照')]
PAGES = ['index.html', 'semantic/index.html', 'instances/index.html',
         'interpretation/index.html', 'elements/index.html', 'label-structure/index.html',
         'quality/index.html', 'city-patterns/index.html']
CITY_FOLLOWUP = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/followup')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ratio(current, total, unit):
    return {'current': current, 'total': total, 'unit': unit} if total else None


def research_progress(rows, corpus):
    """Research deliverables are broader than successful GPU jobs."""
    roadmap = read(Path(__file__).with_name('roadmap.json'))
    tasks = {row['id']: row for row in rows}
    patch_coverage = tasks.get('patch_coverage', {})
    refit = tasks.get('patch_response_refit', {})
    hierarchy = tasks.get('element_hierarchy', {})
    local_compositions = tasks.get('local_compositions', {})
    open_panorama = tasks.get('open_panorama', {})
    coverage = patch_coverage or tasks.get('coverage_trial', {})
    states = {
        'data': ('complete' if tasks['activations']['status'] == 'complete' else 'partial',
                 f'主 SAE：30 城市、89,031 全景；原型层：29 城市、{corpus.get("panoramas", 0):,} 全景。'),
        'units': (tasks['sae']['status'], 'W4096 / K32 训练完成，已有重建曲线、逐 patch 激活和 64 组归纳。'),
        'meaning': ('partial', tasks['dimension_vlm']['detail'] + '；UMAP 和图像证据已上线，独立语义验证未完成。'),
        'element_families': ('partial' if hierarchy else 'pending', hierarchy.get('detail', '待整理维度—要素父类与图像证据。')),
        'local_compositions': ('partial' if local_compositions.get('status') == 'complete' else 'pending',
                               local_compositions.get('detail', '待提取四方向局部要素关系。')),
        'patterns': ('partial' if local_compositions.get('status') == 'complete' else 'pending',
                     local_compositions.get('detail', '尚无跨街景组合候选。')),
        'road_sides': ('pending', '现有组合未按经核查的道路轴线划分同侧和两侧；须补算patch激活与道路几何。'),
        'prototype_capacity': ('pending', '不预设最终原型数；生成、合并、淘汰阈值和容量上限待校准。'),
        'prototype_coverage': ('partial' if coverage.get('status') == 'complete' else coverage.get('status', 'pending'),
                               coverage.get('detail', '待开始逐图patch覆盖试验；语义解释质量另行核查。')),
        'validity': ('partial' if local_compositions.get('status') == 'complete' else 'pending',
                     '已有探索性开发集置换检验；跨种子、人工语义核查和封存测试未完成。'),
        'geography': ('partial', '已有 64 组的城市组成、共现和空间图；新原型的地理归纳尚未完成。'),
        'figure_refresh': ('pending', 'Chapter 5全图重绘及新增原型跨城、同侧/两侧分布图待30城分析完成。'),
        'writing': ('partial', roadmap['writing_evidence'])
    }
    milestones = [{**item, 'status': states[item['id']][0], 'evidence': states[item['id']][1]}
                  for item in roadmap['milestones']]
    for item in milestones:
        if item.get('review_status'):
            if item['review_status'] not in {'partial', 'pending', 'running', 'complete'}:
                raise ValueError(f'Invalid reviewed milestone status: {item["id"]}')
            item['status'] = item['review_status']
        if item['id'] == 'local_compositions' and not local_compositions:
            item['status'] = 'pending'
    focus = '核查跨街景重复的局部要素组合'
    next_step = ('逐例核查遮罩与对象父类，合并方向等价关系；再比较跨种子稳定性、'
                 '三元关系和新组合对完整样本的覆盖。')
    if coverage.get('status') in {'running', 'waiting'}:
        focus = '诊断并改进原型的解释能力'
        next_step = '完成固定模板响应诊断及两组试训，比较开发集质量；人工语义核查和最终测试另行完成。'
    if patch_coverage:
        focus = '让每张图至少80%的有效patch达到解释要求'
        next_step = ('按逐图覆盖缺口训练96个模板，逐图统计是否达标；固定patch NMSE≤0.5，'
                     '同时报告未达标图像、无有效激活patch及全景四方向全部达标比例。语义核查待完成。')
        if patch_coverage['status'] == 'complete':
            next_step = '核查逐图达标比例、失败区域和语义一致性；本轮计算完成不代表所有图像已经达标，测试集继续封存。'
    if refit.get('status') == 'complete':
        focus = '定位景观原型覆盖不足的瓶颈'
        before, after = refit['metrics']['encoder'], refit['metrics']['coverage']
        next_step = (f'冻结96模板的响应优化诊断已完成，{after["images"]}张图平均覆盖'
                     f'{before["mean_image_patch_coverage"]:.1%}→{after["mean_image_patch_coverage"]:.1%}，'
                     f'达标{after["images_meeting_target"]}/{after["images"]}张。'
                     '下一步比较模板数量、秩与空间约束，核查失败区域和语义质量；调参尚未开始，测试集继续封存。')
        for item in milestones:
            if item['id'] == 'prototype_coverage':
                item['evidence'] += ' ' + refit['detail']
    if roadmap.get('active_route') == 'open_panorama':
        if open_panorama.get('status') in {'running', 'preparing'}:
            focus = '构建完整四方向环形全景并开放式发现原型'
            next_step = open_panorama['detail']
        elif open_panorama.get('status') == 'complete':
            focus = '核查开放式全景原型是否稳定且可解释'
            next_step = '检查完整全景mask、跨城市支持、生成合并记录和空间块开发集覆盖；测试封存。'
        elif open_panorama.get('status') == 'error':
            focus = '修复开放式全景原型训练'
            next_step = open_panorama['detail']
        else:
            focus = '构建完整四方向环形全景输入'
            next_step = ('复用现有SAE激活，将0°、90°、180°、270°组成首尾相接的28×112网格；'
                         '随后启动带生成、合并和淘汰的开放式原型训练。')
        for item in milestones:
            if item['id'] == 'patterns':
                item['status'] = 'partial' if open_panorama.get('status') == 'complete' else 'pending'
                item['evidence'] = ('完整四方向原型已生成，语义和稳定性待核查。'
                                    if open_panorama.get('status') == 'complete'
                                    else '完整全景开放式原型尚未形成可发布结果。')
            elif item['id'] == 'prototype_capacity':
                item['evidence'] = ('不预设最终原型数；计算容量只作为上限。原型根据跨全景重复结构生成，'
                                    '相似原型合并，低支持或无验证贡献原型淘汰。')
            elif item['id'] == 'validity':
                item['status'] = 'pending'
                item['evidence'] = '环形旋转、方向打乱、跨城、空间块及跨种子检验待开展。'
            elif item['id'] == 'prototype_coverage':
                metrics = open_panorama.get('metrics') or {}
                if open_panorama.get('status') == 'complete' and metrics:
                    item['status'] = 'partial'
                    item['evidence'] = (
                        f'已评分{metrics["panoramas"]:,}个开发全景；64组池化网格平均重建覆盖'
                        f'{metrics["group_pooled_patch_coverage_all_denominator"]:.1%}，'
                        f'四方向各≥80%网格达标的全景占{metrics["four_directions_each_at_least_80pct"]:.1%}。'
                        '原4096维覆盖与语义解释覆盖尚未评估，不能据此认定目标达成。')
                else:
                    item['status'] = 'pending'
                    item['evidence'] = '开放式全景字典尚未进行逐全景80% patch覆盖评分。'
    if roadmap.get('active_route') == 'local_compositions':
        focus = '核查跨街景重复的局部要素组合'
        next_step = ('人工审阅要素遮罩与对象完整性；合并等价关系，检验跨种子稳定性、'
                     '三元组合以及完整样本上的语义覆盖。')
        for item in milestones:
            if item['id'] == 'local_compositions':
                item['status'] = 'partial' if local_compositions.get('status') == 'complete' else 'pending'
            if item['id'] == 'prototype_capacity':
                item['evidence'] = '候选搜索没有固定槽位；合并等价关系和停止规则仍需校准。'
            if item['id'] == 'validity':
                item['status'] = 'partial' if local_compositions.get('status') == 'complete' else 'pending'
            if item['id'] == 'prototype_coverage':
                item['status'] = 'pending'
                item['evidence'] = '局部要素组合尚未评估逐图80%语义patch覆盖；既有重建试验不适用于当前路线。'
    blocked = [row['label'] + ('等待 GPU 资源。' if row['status'] == 'waiting' else '已暂停，需核查后台任务。' if row['status'] == 'paused' else '运行失败，需处理后继续。')
               for row in rows if row['status'] in {'error', 'paused', 'waiting'}]
    return {'goal': roadmap['goal'], 'focus': focus, 'next_step': next_step,
            'milestones': milestones, 'completed': sum(m['status'] == 'complete' for m in milestones),
            'scope_note': roadmap['scope_note'], 'caveats': roadmap['caveats'],
            'blockers': blocked, 'reviewed_on': roadmap['reviewed_on']}


def snapshot(root=ROOT):
    main = root / 'experiments/vitl30_panorama_w4096_k32'
    proto = root / 'experiments/vitl30_multiscale_prototypes/outputs'
    pipeline = read(proto / 'pipeline_status.json')
    corpus = read(proto / 'corpus.json')
    if corpus and corpus['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Prototype corpus uses a different SAE checkpoint')
    pid = pipeline.get('pid')
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = 'vitl30_multiscale_prototypes.code.run' in Path(f'/proc/{int(pid)}/cmdline').read_text()
        except (OSError, ValueError):
            pass
    stage = pipeline.get('stage')
    terminal = pipeline.get('status') in {'failed', 'stopped'}
    if terminal:
        # The runner's exception handler records "failed"/"stopped" as its
        # stage. Attribute the interruption to the first unfinished dependency.
        if not (proto / 'cache_complete.json').exists():
            stage = 'extract'
        elif not all((proto / 'models' / name / 'complete.json').exists() for name, _ in RUNS):
            stage = 'train'
        elif not (proto / 'evaluation/complete.json').exists():
            stage = 'evaluate'
        else:
            stage = 'interpret'

    def stage_status(name, complete=False):
        if complete:
            return 'complete'
        if stage == name:
            if terminal:
                return 'error' if pipeline['status'] == 'failed' else 'paused'
            if not alive:
                return 'paused'
            return 'waiting' if pipeline.get('status') == 'waiting_for_gpu' else 'running'
        return 'pending'

    rows = []
    training = read(main / 'results/training/training_report.json')
    rows.append({'id': 'sae', 'label': 'BatchTopK SAE',
                 'status': 'complete' if training.get('status') == 'complete' else 'unknown',
                 'detail': f'30 城市 · W4096 / K32 · {training.get("epoch", 0)}/{training.get("epochs", 16)} 轮',
                 'link': '#training'})
    semantics = [main / f'reports/{part}/summary.json' for part in ['semantic_umap', 'instance_umap']]
    semantic_ready = all(p.exists() and read(p).get('checkpoint_sha256') == CHECKPOINT for p in semantics)
    rows.append({'id': 'semantics', 'label': '语义地图与实例',
                 'status': 'complete' if semantic_ready else 'pending',
                 'detail': '维度 UMAP · Top10 实例 · 激活图', 'link': 'semantic/'})
    vlm = read(main / 'reports/vlm_dimensions_top10_masked/summary.json')
    counts = vlm.get('status_counts', {})
    interpreted = counts.get('interpreted', 0)
    unobserved = counts.get('unobserved_in_sample', 0)
    total = vlm.get('dimensions', 4096)
    expansion_dir = main / 'reports/vlm_dimensions_top10_expanded'
    expansion = read(expansion_dir / 'expansion_status.json')
    expanded_summary = read(expansion_dir / 'summary.json')
    ranking = read(main / 'outputs/analysis/interpretation_expanded_top10/ranking_summary.json')
    expansion_alive = False
    if expansion.get('pid'):
        try:
            expansion_alive = 'panorama_vitl30_expand_interpretation' in Path(
                f'/proc/{int(expansion["pid"])}/cmdline').read_text()
        except (OSError, ValueError):
            pass
    vlm_status = 'complete' if interpreted + unobserved == total else 'unknown'
    vlm_detail = f'{interpreted:,} 维已解释；{unobserved:,} 维样本内无正激活'
    vlm_progress = None
    if expansion:
        if ranking and ranking.get('checkpoint_sha256') != CHECKPOINT:
            raise ValueError('Expanded interpretation uses a different SAE checkpoint')
        new_counts = expanded_summary.get('status_counts', {})
        new_done = new_counts.get('interpreted', 0)
        recovered = ranking.get('observed_dimensions', 0)
        no_evidence = ranking.get('unobserved_dimensions', unobserved)
        vlm_detail = (f'{interpreted:,} 维原解释保留；扩样至 {ranking.get("panoramas", 0):,} 全景，'
                      f'新增解释 {new_done}/{recovered} 维；{no_evidence} 维仍无正激活。仅事后解释，含训练图像')
        vlm_status = ('complete' if expansion.get('status') == 'complete' else
                      'error' if expansion.get('status') == 'failed' else
                      'paused' if expansion.get('status') == 'stopped' or not expansion_alive else
                      'waiting' if expansion.get('status') == 'waiting_for_gpu' else 'running')
        if vlm_status == 'waiting':
            vlm_detail += '；等待 GPU，禁用 CPU 推理'
        elif expansion.get('stage') in {'ranking', 'crops', 'sheets'}:
            vlm_detail += '；准备扩样证据'
        vlm_progress = ratio(new_done, recovered, '新增维度解释')
    rows.append({'id': 'dimension_vlm', 'label': '逐维 Qwen 解释',
                 'status': vlm_status, 'detail': vlm_detail, 'progress': vlm_progress,
                 'link': 'interpretation/'})
    cache = read(proto / 'cache_complete.json')
    rows.append({'id': 'activations', 'label': '完整稀疏激活',
                 'status': stage_status('extract', bool(cache)),
                 'detail': f'{cache.get("panoramas", corpus.get("panoramas", 0)):,} 全景 · {len(corpus.get("cities", []))} 城市'})

    runs, completed, steps, total_steps = [], 0, 0, 0
    for name, label in RUNS:
        folder = proto / 'models' / name
        state = read(folder / 'status.json')
        config = read(folder / 'config.json')
        done = bool(read(folder / 'complete.json'))
        epochs = config.get('epochs', state.get('epochs', 8))
        per_epoch = state.get('steps_per_epoch', 9309 if corpus else 0)
        target = epochs * per_epoch
        current = target if done else min(state.get('global_step', 0), target)
        status = stage_status('train', done)
        # A queued run is distinct from the currently active workers.
        if not done and alive and stage == 'train' and name not in pipeline.get('jobs', {}):
            status = 'waiting'
        device = pipeline.get('jobs', {}).get(name, {}).get('device')
        detail = f'{epochs}/{epochs} 轮' if done else f'第 {state.get("epoch", 0)}/{epochs} 轮'
        if device:
            detail += f' · GPU {device.split(":")[-1]}'
        runs.append({'label': label, 'status': status, 'detail': detail,
                     'progress': ratio(current, target, '训练步')})
        completed += int(done)
        steps += current
        total_steps += target
    rows.append({'id': 'prototypes', 'label': '多尺度原型训练',
                 'status': stage_status('train', completed == len(RUNS)),
                 'detail': f'{completed}/4 组训练完成 · 3×3 / 7×7 / 15×15',
                 'progress': ratio(steps, total_steps, '训练步'), 'runs': runs})
    evaluation = proto / 'evaluation'
    eval_done = (evaluation / 'complete.json').exists()
    scored = sum((evaluation / name / 'complete.json').exists() for name, _ in RUNS)
    detail = '完整验证 · 种子稳定性 · NMF · 置换检验'
    if stage == 'evaluate' and not eval_done:
        detail = f'{scored}/4 组完成验证评分；含稳定性与置换检验'
    rows.append({'id': 'evaluation', 'label': '验证与稳定性',
                 'status': stage_status('evaluate', eval_done), 'detail': detail})
    gallery_done = (proto / 'gallery/index.html').exists()
    rows.append({'id': 'gallery', 'label': '原型图像与激活',
                 'status': stage_status('evaluate', gallery_done),
                 'detail': '96 个候选模板的原图、响应及组成证据'})
    interpretation = read(proto / 'interpretation/status.json')
    interpretation_done = (proto / 'interpretation/complete.json').exists()
    detail = '原型图完成后，对候选模板进行解释'
    if interpretation:
        detail = f'已处理 {interpretation.get("processed", 0)}/{interpretation.get("total", 96)} 个候选模板'
    if stage == 'interpret' and pipeline.get('status') == 'waiting_for_gpu':
        detail = '等待可用 GPU 显存后运行 Qwen'
    rows.append({'id': 'prototype_vlm', 'label': '原型 Qwen 解释',
                 'status': stage_status('interpret', interpretation_done), 'detail': detail,
                 'progress': ratio(interpretation.get('processed', 0), interpretation.get('total', 0), '候选模板')})
    coverage = coverage_status(root)
    if coverage:
        rows.append(coverage)
    patch_coverage = patch_coverage_status(root)
    if patch_coverage:
        rows.append(patch_coverage)
    refit = response_refit_status(root)
    if refit:
        rows.append(refit)
    gallery = patch_gallery_status(root)
    if gallery:
        rows.append(gallery)
    rows.extend(relational_status(root))
    if any(row['id'] == 'local_compositions' for row in rows):
        rows = [row for row in rows if row['id'] != 'open_panorama']
    research = research_progress(rows, corpus)
    city = city_pattern_progress(CITY_FOLLOWUP) if root == ROOT else None
    if city:
        rows.extend(city['rows'])
        for milestone in research['milestones']:
            if milestone['id'] in city['milestones']:
                milestone.update(city['milestones'][milestone['id']])
        research.update(focus=city['focus'], next_step=city['next_step'])
        research['completed'] = sum(m['status'] == 'complete' for m in research['milestones'])
    legacy = {'prototypes', 'evaluation', 'gallery', 'prototype_vlm', 'coverage_trial',
              'patch_coverage', 'patch_response_refit', 'patch_prototype_gallery'}
    if city:
        legacy.add('local_compositions')
    rows = [row for row in rows if row['id'] not in legacy]
    research['blockers'] = [row['label'] for row in rows if row['status'] in {'error', 'paused', 'waiting'}]
    return {'schema_version': 1, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'experiment': 'vitl30_panorama_w4096_k32', 'checkpoint_sha256': CHECKPOINT,
            'headline': research['focus'], 'research': research, 'rows': rows,
            'runtime_terminal': (pipeline.get('status') in {'complete', 'failed', 'stopped'} or (bool(pipeline) and not alive))
                and (not expansion or expansion.get('status') in {'complete', 'failed', 'stopped'} or not expansion_alive)
                and (not coverage or coverage['status'] in {'complete', 'error', 'paused', 'pending'})
                and (not patch_coverage or patch_coverage['status'] in {'complete', 'error', 'paused', 'pending'})
                and (not city or city['terminal']),
            'note': ('当前路线从四方向全景中的局部要素遮罩检索开放数量的重复组合。'
                     '计算完成仍不等于语义已验证。')}


def city_pattern_progress(followup):
    """Read bounded queue/artifact summaries; never scan image or feature data."""
    queue = read(followup / 'task_queue.json')
    if not queue:
        return None
    complete = read(followup / 'complete.json')
    if complete and complete['identity']['extract_run_identity'] != queue['run_identity']:
        raise ValueError('City progress extraction identity mismatch')
    labels = {'extract': '30城格网样本冻结推理', 'validate_chunks': '分块质量与身份核对',
              'city_cell_screening': '格网平衡城市汇总', 'candidate_combinations': '跨城共现候选筛选',
              'road_sides': '四方向激活与道路审阅证据',
              'prototype_city_distribution': '候选格网与城市分布', 'chapter5_figures': '论文全图更新与编译'}
    rows = []
    for task in queue['tasks']:
        state = task['status']
        row = {'id': 'city_' + task['id'], 'label': labels.get(task['id'], task['id']),
               'status': {'blocked': 'pending', 'retrying': 'waiting', 'awaiting_script': 'pending'}.get(state, state),
               'detail': task.get('detail', ''), 'link': 'city-patterns/', 'semantic_review': 'pending'}
        if task['id'] == 'extract' and complete:
            row['detail'] = f'{complete["processed"]:,}有效全景 · {complete["valid_cells"]:,}观测格网 · 30城；原SAE保持冻结。'
        if task['id'] == 'road_sides':
            progress = read(followup / 'stages/road_sides/progress.json')
            if 'total' in progress:
                row['progress'] = ratio(progress['processed'], progress['total'], '四方向审阅实例')
            summary = read(followup / 'stages/road_sides/summary.json')
            if summary:
                row['detail'] = f'{summary["examples"]}例证据完成；最终同侧{summary["same_side"]}、两侧{summary["opposite_side"]}、未确认{summary["undetermined"]}。相机标定与人工复核待完成。'
        rows.append(row)
    states = {task['id']: task['status'] for task in queue['tasks']}
    def done(stage):
        return states.get(stage) == 'complete'
    milestones = {}
    if complete:
        milestones['data'] = {'status': 'complete', 'evidence': f'30城空间分析：{complete["processed"]:,}全景，{complete["valid_cells"]:,}格网，每格最多10点；SAE原训练样本89,031全景。'}
    if done('candidate_combinations'):
        summary = read(followup / 'stages/candidate_combinations/summary.json')
        evidence = f'30城共现筛查得到{summary.get("candidate_count", 0)}个候选；22城发现、8城确认，999次双零假设及FDR。要素语义和具体布局未验证。'
        milestones['patterns'] = {'status': 'partial', 'evidence': evidence}
        milestones['local_compositions'] = {'status': 'partial', 'evidence': evidence}
    milestones['road_sides'] = {'status': 'partial' if done('road_sides') else 'running' if states.get('road_sides') == 'running' else 'pending',
        'evidence': '独立道路分割与冻结SAE补算四方向patch遮罩；计算候选不等于确认道路侧别，标定和人工审阅待完成。'}
    if done('prototype_city_distribution'):
        milestones['geography'] = {'status': 'partial', 'evidence': '7个共现候选的全样本格网平衡城市发生率、5km块bootstrap区间和1km格网地图完成；侧别发生率未测量。'}
    if done('chapter5_figures'):
        milestones['figure_refresh'] = {'status': 'complete', 'evidence': '受影响城市图表已重绘，原训练与语义证据保留来源身份；论文编译通过。语义与道路侧别后续需人工核查。'}
        milestones['writing'] = {'status': 'partial', 'evidence': '方法、分母与图注已按新30城空间样本更新并编译；最终原型结论等待语义与道路核查。'}
    pending = next((r for r in rows if r['status'] != 'complete'), None)
    return {'rows': rows, 'milestones': milestones, 'terminal': queue.get('phase') == 'complete',
            'focus': pending['label'] if pending else '核查30城组合的语义与道路布局',
            'next_step': '本地脚本按队列自动衔接；候选分布和论文图表完成后发布。人工确认语义、mask与标定后才统计道路同侧／两侧。'}


def relational_status(root):
    """Publish the element hierarchy and exploratory local-composition result."""
    result = root / 'experiments/vitl30_relational_prototypes/results'
    summary = read(result / 'summary.json')
    pilot = read(result / 'pilot_manifest_summary.json')
    if not summary:
        return []
    if summary['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Element hierarchy uses another SAE checkpoint')
    if pilot and pilot['hierarchy_identity'] != summary['identity']:
        raise ValueError('Relational pilot uses another element hierarchy')
    counts = summary['status_counts']
    rows = [{'id': 'element_hierarchy', 'label': '维度—要素候选表', 'status': 'complete',
             'detail': (f'{summary["dimensions"]:,}维保留；{counts["candidate_ready"]:,}维暂定候选，'
                        f'{len(summary["graph_candidate_family_counts"])}类要素；'
                        f'{summary["family_disagreements"]}维解释父类冲突待核查。人工验证未完成。'),
             'link': 'elements/', 'semantic_review': summary['semantic_validation']}]
    # The published source is the complete-panorama three-node result.  Keep
    # the old path only as a compatibility fallback for isolated snapshot
    # tests; it must never win when the new result is present.
    local_file = root / 'experiments/vitl30_local_compositions/outputs/panorama_motifs/results.json'
    if not local_file.exists():
        local_file = root / 'experiments/vitl30_local_compositions/outputs/results.json'
    if not local_file.exists():
        local_file = root / 'experiments/vitl30_local_compositions/results/results.json'
    local = read(local_file)
    if local:
        if (local.get('checkpoint_sha256') != CHECKPOINT or
                local.get('hierarchy_identity') != summary['identity']):
            raise ValueError('Local-composition result uses another SAE or hierarchy')
        if 'counts' in local:
            detail = (f"{local['sampling']['cities']}城，{local['sampling']['train_panoramas']:,}训练 / "
                      f"{local['sampling']['dev_panoramas']:,}开发全景；"
                      f"{local['counts']['all_observed_keys']:,}个观测关系key，"
                      f"{local['counts']['train_candidates']:,}个三节点候选，"
                      f"网页展示{local['counts']['gallery_patterns']}个布局；遮罩和语义仍待人工核查。")
            status = 'complete' if local.get('status') == 'exploratory' else local.get('status', 'pending')
        else:
            detail = (f"{local['sampling']['cities']}城，{local['sampling']['train_panoramas']:,}训练 / "
                      f"{local['sampling']['dev_panoramas']:,}开发全景；"
                      f"{local['findings']['all_candidates']}个原始关系候选，遮罩和语义仍待人工核查。")
            status = 'complete' if local.get('status') == 'complete' else 'pending'
        rows.append({'id': 'local_compositions', 'label': '全景级多要素关系探索',
                     'status': status, 'detail': detail,
                     'link': 'relations/#local-results', 'semantic_review': 'pending',
                     'test_sealed': True})
        return rows
    open_status = read(root / 'experiments/vitl30_open_panorama_prototypes/outputs/status.json')
    state = {'training':'running','failed':'error'}.get(open_status.get('status'),open_status.get('status','pending'))
    if state not in {'pending', 'preparing', 'running', 'complete', 'error', 'paused'}:
        raise ValueError('Unknown open panorama prototype status')
    detail = open_status.get('detail',
        '待从现有SAE激活构建四方向首尾相接的28×112网格；不预设最终原型数量。')
    if state in {'preparing','running'} and open_status.get('pid'):
        try:
            process = Path(f'/proc/{int(open_status["pid"])}')
            command = (process/'cmdline').read_text()
            if 'vitl30_open_panorama_prototypes' not in command:
                raise OSError('Runner missing')
        except (OSError,ValueError):
            state = 'paused'
            detail = '进程已退出，检查点保留；'+detail
    rows.append({'id': 'open_panorama', 'label': '开放式四方向全景原型', 'status': state,
                 'detail': detail, 'link': 'elements/#open-method',
                 'semantic_review': 'pending', 'test_sealed': True,
                 'metrics': open_status.get('metrics')})
    return rows


def coverage_status(root):
    """Read the coverage pilot without importing models or touching activation data."""
    out = root / 'experiments/vitl30_prototype_coverage/outputs'
    protocol = read(out / 'protocol.json')
    if not protocol:
        return None
    pipeline = read(out / 'pipeline_status.json')
    status = pipeline.get('status', 'pending')
    alive = False
    if pipeline.get('pid'):
        try:
            alive = 'vitl30_prototype_coverage.code.run' in Path(f'/proc/{int(pipeline["pid"])}/cmdline').read_text()
        except (OSError, ValueError):
            pass
    state = ('complete' if status == 'complete' else 'error' if status == 'failed' else
             'paused' if status == 'stopped' or (pipeline and not alive) else
             'waiting' if status in {'waiting_for_gpu', 'retrying_gpu_memory'} else
             'running' if status == 'running' else 'pending')
    counts = protocol['split_counts']
    detail = (f'{counts["train"]:,}训练 / {counts["dev"]:,}开发 / {counts["test"]:,}测试全景；'
              '每尺度32模板；测试封存。')
    stage, arm = pipeline.get('stage'), pipeline.get('arm')
    if stage == 'prepare':
        prep = read(out / 'preparation_status.json')
        detail += f' 训练集校准 {prep.get("panoramas", 0):,}/{protocol["panoramas"]:,}。'
    elif stage == 'diagnostic':
        diag = read(out / 'diagnostic/status.json')
        detail += f' 固定模板响应诊断 {diag.get("views", 0)}/{diag.get("total", 116)}方向图。'
        if 'refit_mean_nmse' in diag:
            detail += f' NMSE {diag["before_mean_nmse"]:.3f}→{diag["refit_mean_nmse"]:.3f}（无稀疏惩罚的探索诊断）。'
    elif stage == 'train':
        training = read(out / 'models' / arm / 'status.json')
        detail += f' {arm} 第{training.get("epoch", 0)}/4轮，{training.get("global_step", 0):,}训练步。'
    elif stage == 'evaluate':
        evaluation = read(out / 'development' / arm / 'status.json')
        detail += f' {arm} 开发集评分 {evaluation.get("cities", 0)}/29城市。'
    elif status == 'complete':
        detail += ' 两组试训与开发评分完成；人工核查未完成，80%语义覆盖尚未验证。'
    markers = [out / 'calibration.json', out / 'diagnostic/complete.json']
    runs = []
    for name, label in [('balanced_reference', '原重建目标'), ('energy_warmup', '能量目标＋稀疏预热')]:
        train = read(out / 'models' / name / 'status.json')
        done = (out / 'models' / name / 'complete.json').exists()
        per_epoch = train.get('steps_per_epoch', 8178)
        runs.append({'label': label, 'status': 'complete' if done else state if stage == 'train' and arm == name else 'pending',
                     'detail': '4轮筛选试训',
                     'progress': ratio(4 * per_epoch if done else train.get('global_step', 0), 4 * per_epoch, '训练步')})
        markers += [out / 'models' / name / 'complete.json', out / 'development' / name / 'complete.json']
    markers.append(out.parent / 'results/pilot_summary.json')
    return {'id': 'coverage_trial', 'label': '重建目标对照试训', 'status': state, 'detail': detail,
            'progress': ratio(sum(p.exists() for p in markers), len(markers), '计算阶段'), 'runs': runs}


def patch_coverage_status(root):
    """Actual per-image coverage run; no activation scans or CUDA imports."""
    out = root / 'experiments/vitl30_patch_coverage/outputs'
    protocol = read(out / 'protocol.json')
    if not protocol:
        return None
    pipeline = read(out / 'pipeline_status.json')
    status = pipeline.get('status', 'pending')
    alive = False
    if pipeline.get('pid'):
        try:
            alive = 'vitl30_patch_coverage.code.run' in Path(f'/proc/{int(pipeline["pid"])}/cmdline').read_text()
        except (OSError, ValueError):
            pass
    state = ('complete' if status == 'complete' else 'error' if status == 'failed' else
             'paused' if status == 'stopped' or (pipeline and not alive) else
             'waiting' if status in {'waiting_for_gpu', 'retrying_gpu_memory'} else
             'running' if status == 'running' else 'pending')
    counts, policy = protocol['split_counts'], protocol['policy']
    detail = (f'目标：每张方向图≥{policy["target_fraction"]:.0%}有效patch达标；'
              f'固定patch NMSE≤{policy["maximum_patch_nmse"]:g}（重建判据，语义待核查）。'
              f' {counts["train"] * 4:,}训练图 / {counts["dev"] * 4:,}开发图；'
              f'{counts["test"] * 4:,}测试图封存。')
    training = read(out / 'model/status.json')
    stage = pipeline.get('stage')
    if stage == 'train':
        detail += (f' GPU {pipeline.get("device", "cuda:?").split(":")[-1]} · '
                   f'第{training.get("epoch", 0)}/{protocol["epochs"]}轮，'
                   f'{training.get("global_step", 0):,}训练步。')
    elif stage in {'evaluate_dev', 'evaluate_train'}:
        split = stage.removeprefix('evaluate_')
        scoring = read(out / 'evaluation' / split / 'status.json')
        detail += f' {"开发" if split == "dev" else "训练"}集逐图评分 {scoring.get("cities", 0)}/{protocol["cities"]}城市。'
    epochs = sorted((out / 'model').glob('epoch_*.json'))
    if epochs:
        recent = read(epochs[-1])
        detail += (f' 第{recent["epoch"]}轮固定开发监控：达标图{recent["image_pass_fraction"]:.1%}，'
                   f'平均patch覆盖{recent["mean_image_patch_coverage"]:.1%}，'
                   f'最差图{recent["minimum_image_patch_coverage"]:.1%}。')
    metrics = {}
    for split, name in [('dev', '开发'), ('train', '训练')]:
        summary = read(out / 'evaluation' / split / 'complete.json')
        if summary:
            keys = ['images', 'image_pass_fraction', 'mean_image_patch_coverage',
                    'panorama_all_four_pass_fraction', 'images_below_target',
                    'invalid_patches', 'all_images_meet_target', 'coverage_quantiles']
            metrics[split] = {k: summary[k] for k in keys}
            detail += (f' 完整{name}集：平均patch覆盖{summary["mean_image_patch_coverage"]:.1%}，'
                       f'达标图{summary["image_pass_fraction"]:.3%}，'
                       f'四方向全部达标全景{summary["panorama_all_four_pass_fraction"]:.1%}。')
    done = (out / 'model/complete.json').exists()
    per_epoch = training.get('steps_per_epoch', 8178)
    total = protocol['epochs'] * per_epoch
    return {'id': 'patch_coverage', 'label': '逐图80% patch覆盖', 'status': state, 'detail': detail,
            'link': 'prototypes/#full-evaluation',
            'policy': policy, 'semantic_review': 'pending', 'test_sealed': True, 'metrics': metrics,
            'progress': ratio(total if done else training.get('global_step', 0), total, '训练步'),
            'runs': [{'label': f'完整{label}集评分',
                      'status': 'complete' if split in metrics else state if stage == f'evaluate_{split}' else 'pending',
                      'detail': f'{counts[split] * 4:,}张方向图；保存未达标清单和逐patch覆盖图'}
                     for split, label in [('dev', '开发'), ('train', '训练')]]}


def response_refit_status(root):
    out = root / 'experiments/vitl30_patch_coverage/outputs'
    complete = read(out / 'response_refit_v1/complete.json')
    if not complete:
        return None
    protocol = read(out / 'protocol.json')
    if complete['protocol']['experiment_identity'] != protocol.get('identity'):
        raise ValueError('Response diagnostic belongs to another coverage protocol')
    if complete['test_used'] or not complete['protocol']['test_sealed']:
        raise ValueError('Response diagnostic unexpectedly used the sealed test')
    before = complete['summaries']['encoder']
    after = complete['summaries']['coverage']
    detail = (f'{before["panoramas"]}城各1开发全景，{before["images"]}张方向图；冻结96模板，'
              f'平均patch覆盖{before["mean_image_patch_coverage"]:.1%}→{after["mean_image_patch_coverage"]:.1%}，'
              f'{after["images_meeting_target"]}/{after["images"]}张达到80%。'
              '150步响应优化，约50步后趋稳；有限步诊断不是理论上限，语义待核查。')
    return {'id': 'patch_response_refit', 'label': '冻结模板响应诊断', 'status': 'complete',
            'detail': detail, 'link': 'prototypes/#response-refit',
            'test_sealed': True, 'semantic_review': 'pending',
            'metrics': complete['summaries']}


def patch_gallery_status(root):
    out = root / 'experiments/vitl30_patch_coverage/outputs'
    complete = read(out / 'gallery/complete.json')
    if not complete:
        return None
    protocol = read(out / 'protocol.json')
    identity = complete['identity']
    if (identity['experiment_identity'] != protocol.get('identity') or
            identity['split'] != 'dev' or identity['test_used']):
        raise ValueError('Current prototype gallery belongs to another experiment or partition')
    missing = complete['templates'] - complete['templates_with_examples']
    return {'id': 'patch_prototype_gallery', 'label': '当前原型逐模板实例', 'status': 'complete',
            'detail': (f'{identity["images"]:,}开发图检索完成；{complete["templates_with_examples"]}个有实例模板，'
                       f'共{complete["examples"]}组原图、贡献热图、裁剪；{missing}个模板暂无实例。语义一致性待核查。'),
            'link': 'prototypes/#prototype-gallery', 'semantic_review': 'pending', 'test_sealed': True,
            'model_sha256': identity['model_sha256']}


def write_snapshot(site, value):
    path = site / 'data/project-progress.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def install(site):
    """Install on the public pages; keep feature iframes uncluttered."""
    for name in ['project-progress.css', 'project-progress.js']:
        target = site / 'assets' / name
        target.write_bytes((TEMPLATES / name).read_bytes())
    for name in PAGES:
        path = site / name
        if not path.exists():
            continue
        doc = path.read_text()
        if 'id="project-progress"' in doc:
            continue
        prefix = '' if name == 'index.html' else '../'
        nav_end = doc.index('</nav>') + len('</nav>')
        before, content = doc[:nav_end], doc[nav_end:]
        before = before.replace('<nav class="site-nav"', f'<link rel="stylesheet" href="{prefix}assets/project-progress.css?v=1"><nav class="site-nav"', 1)
        ends = [i for token in ['<script', '</body>', '</html>'] if (i := content.find(token)) >= 0]
        stop = min(ends)
        kind = ' project-map' if name.startswith('semantic/') else ''
        sidebar = '''<aside class="project-sidebar" aria-label="项目进度">
<details id="project-progress" open><summary><span class="project-progress-kicker">PROJECT STATUS</span><span class="project-progress-heading">项目进度<span class="project-progress-pulse" aria-hidden="true"></span></span><span id="project-progress-headline">读取最新进度…</span></summary>
<div class="project-progress-content"><p id="project-research-goal"></p>
<div class="project-focus"><span class="project-progress-kicker">当前关键路径</span><strong id="project-research-focus"></strong><p id="project-next-step"></p></div>
<p class="project-progress-context" id="project-milestone-count"></p><div id="project-milestones"></div>
<div id="project-blockers"></div>
<details class="project-runtime"><summary>计算任务明细<span id="project-runtime-count"></span></summary>
<table class="project-progress-table"><caption class="project-sr-only">当前主实验与景观原型的阶段进度</caption><thead class="project-sr-only"><tr><th scope="col">阶段与进度</th><th scope="col">状态</th></tr></thead><tbody id="project-progress-rows"></tbody></table>
<p id="project-progress-note"></p></details><details class="project-scope"><summary>研究范围与待核查项</summary><div id="project-research-scope"></div></details>
<div class="project-progress-bottom"><time id="project-progress-time"></time><button type="button" id="project-progress-refresh" aria-label="刷新项目进度">刷新</button></div><p id="project-progress-freshness" role="status">读取发布快照…</p>
<noscript>启用 JavaScript 后显示进度表，或查看 <a href="@PREFIX@data/project-progress.json">进度数据</a>。</noscript></div></details></aside>'''.replace('@PREFIX@', prefix)
        path.write_text(before + f'<div class="project-layout{kind}"><div class="project-main">' + content[:stop] + '</div>' + sidebar + '</div>' + f'<script src="{prefix}assets/project-progress.js?v=1" defer></script>' + content[stop:])
    write_snapshot(site, snapshot())


def publish(checkout):
    """Commit only the JSON; stop if a newer site was published elsewhere."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    git('fetch', '--quiet', 'origin', 'gh-pages')
    if git('rev-parse', 'HEAD') != git('rev-parse', 'origin/gh-pages'):
        raise RuntimeError('Publishing checkout differs from remote; refresh it before continuing')
    value = snapshot()
    path = checkout / 'data/project-progress.json'
    previous = read(path)
    # A timestamp alone is not a reason to trigger another Pages deployment.
    if {k: v for k, v in value.items() if k != 'updated_utc'} == {k: v for k, v in previous.items() if k != 'updated_utc'}:
        return value, False
    write_snapshot(checkout, value)
    git('add', '--sparse', '--', 'data/project-progress.json')
    git('commit', '--only', '-m', 'Update main experiment progress', '--', 'data/project-progress.json')
    git('push', 'origin', 'HEAD:gh-pages')
    return value, True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--install', action='store_true', help='Add the sidebar to an existing built bundle')
    parser.add_argument('--publish', action='store_true', help='Commit/push only progress JSON in the Pages checkout')
    parser.add_argument('--watch', action='store_true', help='Refresh until the pipeline finishes or stops')
    parser.add_argument('--interval', type=int, default=600)
    args = parser.parse_args()
    if args.install:
        install(args.site_dir)
    lock = None
    if args.watch:
        lock = Path('/tmp/usp-site-progress-publisher.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        if args.publish:
            value, changed = publish(args.site_dir)
        else:
            value, changed = snapshot(), True
            write_snapshot(args.site_dir, value)
        print(json.dumps({'updated_utc': value['updated_utc'], 'published': changed,
                          'headline': value['headline']}, ensure_ascii=False), flush=True)
        if not args.watch or value['runtime_terminal']:
            break
        # Short waits allow prompt shutdown without a long blocking tool call.
        for _ in range(max(60, args.interval) // 15):
            time.sleep(15)
