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
         'interpretation/index.html', 'combinations/index.html']


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ratio(current, total, unit):
    return {'current': current, 'total': total, 'unit': unit} if total else None


def research_progress(rows, corpus):
    """Research deliverables are broader than successful GPU jobs."""
    roadmap = read(Path(__file__).with_name('roadmap.json'))
    tasks = {row['id']: row for row in rows}
    states = {
        'data': ('complete' if tasks['activations']['status'] == 'complete' else 'partial',
                 f'主 SAE：30 城市、89,031 全景；原型层：29 城市、{corpus.get("panoramas", 0):,} 全景。'),
        'units': (tasks['sae']['status'], 'W4096 / K32 训练完成，已有重建曲线、逐 patch 激活和 64 组归纳。'),
        'meaning': ('partial', tasks['dimension_vlm']['detail'] + '；UMAP 和图像证据已上线，独立语义验证未完成。'),
        'patterns': (tasks['prototype_vlm']['status'] if tasks['prototype_vlm']['status'] == 'complete' else tasks['prototypes']['status'],
                     tasks['prototypes']['detail'] + '；原型图与解释' + ('已完成。' if tasks['prototype_vlm']['status'] == 'complete' else '待补齐。')),
        'validity': (tasks['evaluation']['status'], tasks['evaluation']['detail']),
        'geography': ('partial', '已有 64 组的城市组成、共现和空间图；新原型的地理归纳尚未完成。'),
        'writing': ('partial', roadmap['writing_evidence'])
    }
    # Training alone does not finish the prototype discovery deliverable.
    if states['patterns'][0] == 'complete' and tasks['prototype_vlm']['status'] != 'complete':
        states['patterns'] = ('partial', states['patterns'][1])
    milestones = [{**item, 'status': states[item['id']][0], 'evidence': states[item['id']][1]}
                  for item in roadmap['milestones']]
    if tasks['prototypes']['status'] != 'complete':
        focus = '完成原型模型与打乱对照'
        next_step = '四组模型完成后，自动进入完整验证、稳定性分析和实例图生成。'
    elif tasks['evaluation']['status'] != 'complete':
        focus = '检验原型是否稳定、有结构意义'
        next_step = '完成全部验证、种子匹配、NMF 和置换检验，核对真实结构相对对照的表现。'
    elif tasks['prototype_vlm']['status'] != 'complete':
        focus = '建立可核查的原型证据库'
        next_step = '补齐原图、贡献激活图和 Qwen 候选解释，检查混合与无效模板。'
    else:
        focus = '从计算结果形成研究结论'
        next_step = '核查单元和原型语义、整理城市证据，将支持的结论与局限写入 Chapter 5。'
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
    rows.append({'id': 'dimension_vlm', 'label': '逐维 Qwen 解释',
                 'status': 'complete' if interpreted + unobserved == total else 'unknown',
                 'detail': f'{interpreted:,} 维已解释；{unobserved:,} 维样本内无正激活',
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
    research = research_progress(rows, corpus)
    return {'schema_version': 1, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'experiment': 'vitl30_panorama_w4096_k32', 'checkpoint_sha256': CHECKPOINT,
            'headline': research['focus'], 'research': research, 'rows': rows,
            'runtime_terminal': pipeline.get('status') in {'complete', 'failed', 'stopped'} or (bool(pipeline) and not alive),
            'note': '各进度条仅表示该阶段已完成的计算量。96 是候选模板容量，解释名称仍需核查。'}


def write_snapshot(site, value):
    path = site / 'data/project-progress.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def install(site):
    """Install once on the five public pages; keep feature iframes uncluttered."""
    for name in ['project-progress.css', 'project-progress.js']:
        target = site / 'assets' / name
        target.write_bytes((TEMPLATES / name).read_bytes())
    for name in PAGES:
        path = site / name
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
