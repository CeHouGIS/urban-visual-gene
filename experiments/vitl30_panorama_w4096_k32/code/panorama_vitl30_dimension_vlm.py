"""Describe individual SAE dimensions from their existing Top100 clean crops.

The VLM sees every available Top100 crop, in rank-labelled contact sheets. It
does not receive decoder groups, geography, embedding coherence or heatmaps.
Descriptions are model hypotheses tied to these examples, not semantic truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageOps

from formal.analysis.semantic_feature_umap import digest
from sae_experiments.panorama_vitl30 import OUTPUT, WIDTH

SOURCE = OUTPUT / 'analysis/semantic_umap_top100'
REPORT = Path('/root/code/urban-visual-gene/reports/vitl30_dimension_vlm')
MODEL = Path('/root/.cache/modelscope/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master')
TYPES = {'object', 'object_part', 'vegetation', 'built_structure', 'surface_texture',
         'color_light', 'scene_context', 'camera_artifact', 'mixed', 'unclear'}
CONSISTENCY = {'mostly_consistent', 'mixed', 'unclear'}
PROMPT = '''解释一个街景SAE维度。全部拼图都属于同一个维度，每格是不同全景的高激活裁剪，数字仅为排名。比较全部例子，找占主要比例的共同可见内容，给出具体中文名称。相同对象在颜色、角度、光照上的变化不算语义混合。只有多个互不相关的内容都占明显比例、找不到主要共同内容时才用mixed；看不清用unclear。允许维度只表示纹理、颜色、光照或场景上下文，不要强行解释为物体。不推断城市、地理、品牌或不可读文字。灰色留白和边界是拼图排版。结论只适用于这些高激活裁剪。
只输出JSON，必须同时包含全部7个字段：label、type、cues、alternatives、consistency、confidence、caveats。type字段不可省略。
type从这些值中选一个：object, object_part, vegetation, built_structure, surface_texture, color_light, scene_context, camera_artifact, mixed, unclear。
consistency从mostly_consistent, mixed, unclear中选一个。confidence从low, medium, high中选一个。
label是具体中文候选含义。cues是共同可见依据。alternatives是实际异类或其他可能解释，没有就用空列表。caveats是解释局限。三个列表各最多2条短句。confidence是模型主观信心，不是准确率。中文总计不超过140字。
下面只演示字段格式，内容请按实际图像填写：
{"label":"车轮及轮胎", "type":"object_part", "cues":["重复出现车轮轮廓"], "alternatives":[], "consistency":"mostly_consistent", "confidence":"medium", "caveats":["部分区域模糊"]}'''


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def parse_response(raw, truncated=False):
    if truncated:
        raise ValueError('truncated generation')
    start, stop = raw.find('{'), raw.rfind('}')
    if start < 0 or stop < start:
        raise ValueError('missing JSON')
    row = json.loads(raw[start:stop + 1])
    if not isinstance(row, dict) or set(row) != {'label', 'type', 'cues', 'alternatives', 'consistency', 'confidence', 'caveats'}:
        raise ValueError('incorrect keys')
    if not isinstance(row['label'], str) or not 1 <= len(row['label']) <= 100:
        raise ValueError('invalid label')
    if row['type'] not in TYPES or row['consistency'] not in CONSISTENCY or row['confidence'] not in {'low', 'medium', 'high'}:
        raise ValueError('invalid enum')
    for name in ['cues', 'alternatives', 'caveats']:
        values = row[name]
        if not isinstance(values, list) or len(values) > 2 or any(not isinstance(x, str) or len(x) > 200 for x in values):
            raise ValueError(f'invalid {name}')
    return row


def contact_sheet(source, rows, target, tile=128, columns=10):
    """Every crop retains its aspect ratio. Ranks are the only input labels."""
    if not rows:
        raise ValueError('empty sheet')
    height = ((len(rows) + columns - 1) // columns) * tile
    panel = Image.new('RGB', (columns * tile, height), '#d8d8d8')
    draw = ImageDraw.Draw(panel)
    for index, row in enumerate(rows):
        x, y = index % columns * tile, index // columns * tile
        with Image.open(source / row['crop']) as original:
            crop = ImageOps.contain(original.convert('RGB'), (tile - 8, tile - 20), Image.Resampling.BICUBIC)
        panel.paste(crop, (x + (tile - crop.width) // 2, y + 16 + (tile - 20 - crop.height) // 2))
        draw.text((x + 4, y + 1), str(row['rank']), fill='#111111')
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.save(target, quality=90)
    return panel.size


def prepare(args):
    args.report.mkdir(parents=True, exist_ok=True)
    source_run = json.loads((args.source / 'run.json').read_text())
    manifest_hash = digest(args.source / 'crops.jsonl')
    weights = {p.name: digest(p) for p in sorted(args.model.glob('model*.safetensors'))}
    if not weights:
        raise ValueError('No local VLM weights')
    identity = {'version': 1, 'checkpoint_sha256': source_run['checkpoint_sha256'],
                'crop_manifest_sha256': manifest_hash, 'model': str(args.model.resolve()),
                'model_config_sha256': digest(args.model / 'config.json'), 'model_weights_sha256': weights,
                'prompt': PROMPT, 'tile_size': 128, 'columns': 10, 'crops_per_sheet': 50,
                'max_new_tokens': args.max_new_tokens, 'precision': 'bfloat16',
                'generation': {'do_sample': False, 'repetition_penalty': 1.03},
                'code_sha256': digest(Path(__file__))}
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = args.report / 'identity.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError('VLM run identity changed; choose a new report directory')
    atomic_json(target, identity)
    groups = defaultdict(list)
    for observation, line in enumerate((args.source / 'crops.jsonl').open()):
        row = json.loads(line)
        groups[int(row['feature'])].append({'observation_index': observation, 'rank': int(row['rank']),
            'crop': row['crop'], 'city': row['city'], 'pano_index': int(row['pano_index'])})
    evidence = []
    for feature in range(WIDTH):
        rows = sorted(groups[feature], key=lambda r: r['rank'])
        if len({r['pano_index'] for r in rows}) != len(rows):
            raise ValueError('Top100 must contain distinct panoramas')
        sheets = [f'inputs/f{feature:04d}_{i // 50}.jpg' for i in range(0, len(rows), 50)]
        evidence.append({'feature': feature, 'count': len(rows), 'crops': rows, 'sheets': sheets})
    atomic_json(args.report / 'evidence.json', {'identity': identity['identity'], 'dimensions': evidence})
    (args.report / 'records').mkdir(exist_ok=True)
    for row in evidence:
        if not row['count']:
            atomic_json(args.report / 'records' / f'f{row["feature"]:04d}.json', {
                'feature': row['feature'], 'identity': identity['identity'], 'status': 'unobserved_in_sample',
                'crop_count': 0, 'interpretation': None, 'independently_validated': False,
                'reason': 'No positive activation in the source sample; meaning unknown'})
    print('PREPARED', len(evidence), 'dimensions;', sum(bool(r['count']) for r in evidence), 'with crops', flush=True)


def reusable(record, identity):
    if record.get('identity') != identity or record.get('status') != 'interpreted':
        return False
    try:
        return parse_response(record['raw_response'], record.get('truncated', False)) == record['interpretation']
    except (ValueError, TypeError, KeyError):
        return False


def worker(args):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    torch.manual_seed(42)
    identity = json.loads((args.report / 'identity.json').read_text())
    if identity['code_sha256'] != digest(Path(__file__)):
        raise ValueError('Worker code changed')
    manifest = json.loads((args.report / 'evidence.json').read_text())
    if manifest['identity'] != identity['identity']:
        raise ValueError('Evidence identity mismatch')
    requested = None if not args.features else set(map(int, args.features.split(',')))
    selected = [r for r in manifest['dimensions'] if r['count'] and r['feature'] % args.shards == args.shard
                and (requested is None or r['feature'] in requested)]
    pending = []
    for row in selected:
        target = args.report / 'records' / f'f{row["feature"]:04d}.json'
        if target.exists() and reusable(json.loads(target.read_text()), identity['identity']):
            continue
        pending.append(row)
    if not pending:
        print('NOTHING PENDING', args.shard, flush=True)
        return
    device = torch.device(args.device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    runtime = {'model_class': type(model).__name__, 'device': str(device), 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name(device), 'batch_size': args.batch_size,
               'updated_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(args.report / f'runtime_shard{args.shard}.json', runtime)

    def run_batch(batch, attempt=0):
        conversations = []
        for row in batch:
            content = []
            for start, relative in zip(range(0, row['count'], 50), row['sheets']):
                path = args.report / relative
                if not path.exists():
                    contact_sheet(args.source, row['crops'][start:start + 50], path)
                with Image.open(path) as sheet:
                    content.append({'type': 'image', 'image': sheet.convert('RGB')})
            prompt = PROMPT + f'\n本次共{row["count"]}个不同全景的裁剪。'
            if row['count'] < 20:
                prompt += ' 实例较少，只提出暂定描述；confidence必须为low，caveats说明样本不足。'
            if attempt:
                previous_path = args.report / 'records' / f'f{row["feature"]:04d}.json'
                previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
                error = (previous.get('attempts') or [{}])[-1].get('error', 'invalid JSON')
                prompt += f' 上次输出未通过格式检查：{error}。重新检查全部7个字段，尤其不要漏掉type字段；不要输出Markdown。'
            content.append({'type': 'text', 'text': prompt})
            conversations.append([{'role': 'user', 'content': content}])
        inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
            padding=True, return_dict=True, return_tensors='pt',
            processor_kwargs={'images_kwargs': {'do_resize': False}})
        inputs = inputs.to(device)
        begin = time.monotonic()
        limit = args.max_new_tokens + 64 * attempt
        try:
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                    repetition_penalty=1.03)
        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            run_batch(batch[:middle], attempt)
            run_batch(batch[middle:], attempt)
            return
        elapsed = time.monotonic() - begin
        generated = tokens[:, inputs['input_ids'].shape[1]:]
        raw_responses = processor.batch_decode(generated, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        retry = []
        for row, raw, emitted in zip(batch, raw_responses, generated):
            truncated = not any(int(t) in eos for t in emitted) and len(emitted) >= limit
            target = args.report / 'records' / f'f{row["feature"]:04d}.json'
            previous = json.loads(target.read_text()) if target.exists() else {}
            result = {'feature': row['feature'], 'identity': identity['identity'], 'crop_count': row['count'],
                      'input_sheets': row['sheets'], 'raw_response': raw, 'truncated': truncated,
                      'independently_validated': False, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                      'evidence_scope': 'all available Top100 crop observations; not all feature activations'}
            outcome = {'attempt': attempt + 1, 'raw_response': raw, 'truncated': truncated,
                       'batch_seconds': elapsed, 'batch_size': len(batch)}
            try:
                interpretation = parse_response(raw, truncated)
                if row['count'] < 20 and interpretation['confidence'] != 'low':
                    raise ValueError('small sample must have low confidence')
                result.update(status='interpreted', interpretation=interpretation)
                outcome['schema_valid'] = True
            except (ValueError, TypeError) as exc:
                result.update(status='response_invalid', interpretation=None)
                outcome.update(schema_valid=False, error=str(exc))
                if attempt < 2:
                    retry.append(row)
            result['attempts'] = previous.get('attempts', []) + [outcome]
            atomic_json(target, result)
        print('VLM_BATCH', 'shard', args.shard, 'features', [r['feature'] for r in batch],
              'seconds', round(elapsed, 2), 'retry', len(retry), flush=True)
        del inputs, tokens, generated
        if retry:
            run_batch(retry, attempt + 1)

    begin = time.monotonic()
    for start in range(0, len(pending), args.batch_size):
        run_batch(pending[start:start + args.batch_size])
        print('PROGRESS', args.shard, min(start + args.batch_size, len(pending)), '/', len(pending),
              'elapsed', round(time.monotonic() - begin, 1), flush=True)


def report(args):
    identity = json.loads((args.report / 'identity.json').read_text())
    evidence = json.loads((args.report / 'evidence.json').read_text())['dimensions']
    audit_path = Path('/root/code/urban-visual-gene/reports/vitl30_unit_motifs/dimension_audit.json')
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    rows, complete, data = [], 0, []
    for feature, item in enumerate(evidence):
        path = args.report / 'records' / f'f{feature:04d}.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        if record and record['identity'] != identity['identity']:
            raise ValueError('Stale record')
        annotation = record.get('interpretation') or {}
        status = record.get('status', 'pending')
        complete += status in {'interpreted', 'unobserved_in_sample'}
        row = {'feature': feature, 'status': status, 'crop_count': item['count'],
               'label_zh': annotation.get('label', ''), 'semantic_type': annotation.get('type', ''),
               'consistency': annotation.get('consistency', ''), 'confidence_self_reported': annotation.get('confidence', ''),
               'cues_zh': annotation.get('cues', []), 'alternatives_zh': annotation.get('alternatives', []),
               'caveats_zh': annotation.get('caveats', []), 'independently_validated': False,
               'embedding_evidence_status': audit[feature]['status'] if audit else '',
               'original_evidence': str((args.source / 'features' / f'{feature}.html').resolve()) if item['count'] else '',
               'record': f'records/f{feature:04d}.json'}
        rows.append(row)
        data.append({**row, 'sheets': item['sheets'] if status == 'interpreted' else []})
    atomic_json(args.report / 'dimension_interpretations.json', {'identity': identity['identity'], 'dimensions': rows})
    target = args.report / 'dimension_interpretations.csv'
    with target.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    summary = {'dimensions': WIDTH, 'completed_or_explicitly_unobserved': complete,
               'status_counts': dict(Counter(r['status'] for r in rows)),
               'semantic_type_counts': dict(Counter(r['semantic_type'] for r in rows if r['status'] == 'interpreted')),
               'consistency_counts': dict(Counter(r['consistency'] for r in rows if r['status'] == 'interpreted')),
               'model': str(args.model), 'independently_validated': False,
               'updated_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(args.report / 'summary.json', summary)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐维 VLM 含义记录</title><style>
body{font:15px system-ui;background:#f3f5f7;color:#243344;max-width:1250px;margin:auto;padding:24px}input,select,button{font:inherit;padding:9px;border:1px solid #acb8c4;border-radius:6px;background:white}header,section{background:white;padding:20px;border-radius:10px;margin-bottom:18px}.controls{display:flex;gap:8px;flex-wrap:wrap}p{line-height:1.7}a{color:#17698d}#listing{max-height:350px;overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #eee;text-align:left}tr[data-feature]{cursor:pointer}tr[data-feature]:hover{background:#e6f3ff}img{width:100%;height:auto}pre{white-space:pre-wrap}.muted{color:#627080}</style>
<header><h1>4,096 个 SAE 维度：VLM 含义记录</h1><p>直接使用已有 Top100 干净激活区域，逐维输入 Qwen3-VL-2B。全部可用裁剪按排名展示；灰色留白和编号只是排版。名称是 VLM 候选解释，confidence 是模型自述，均未经独立语义验证。无激活维度不编造含义。</p><p><a href="dimension_interpretations.csv">CSV</a> · <a href="dimension_interpretations.json">JSON</a> · <a href="REPORT.md">说明</a></p><p id="progress"></p></header><section><div class="controls"><input id="search" placeholder="搜索编号、名称或类型"><select id="type"><option value="">全部类型</option></select></div><div id="listing"></div></section><section id="detail"><h2>选择一个维度</h2></section>
'''
    page += '<script id="data" type="application/json">' + json.dumps(data, ensure_ascii=False).replace('<', '\\u003c') + '</script>'
    page += '<script>const data=JSON.parse(document.querySelector("#data").textContent);'
    page += f'document.querySelector("#progress").textContent="已记录或明确无实例：{complete} / 4096";'
    page += '''const types=document.querySelector('#type');[...new Set(data.map(d=>d.semantic_type).filter(Boolean))].sort().forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;types.append(o)});
function listing(){const q=document.querySelector('#search').value.trim().toLowerCase();const subset=data.filter(d=>(!types.value||d.semantic_type===types.value)&&(!q||JSON.stringify([d.feature,d.label_zh,d.semantic_type]).toLowerCase().includes(q)));const table=document.createElement('table');table.innerHTML='<tr><th>维度</th><th>VLM候选含义</th><th>类型</th><th>实例数</th><th>状态</th></tr>';subset.forEach(d=>{const tr=document.createElement('tr');tr.dataset.feature=d.feature;[d.feature,d.label_zh||'—',d.semantic_type,d.crop_count,d.status].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});tr.onclick=()=>{location.hash='f'+d.feature};table.append(tr)});document.querySelector('#listing').replaceChildren(table)}
function detail(){const m=location.hash.match(/^#f(\\d+)$/);if(!m)return;const d=data[Number(m[1])];if(!d)return;const box=document.querySelector('#detail');box.replaceChildren();const h=document.createElement('h2');h.textContent=`f${d.feature} · ${d.label_zh||d.status}`;box.append(h);const p=document.createElement('p');p.textContent=`${d.crop_count} 个裁剪；${d.semantic_type}；一致性：${d.consistency}；模型自述信心：${d.confidence_self_reported}。`;box.append(p);[['共同依据',d.cues_zh],['异类/其他解释',d.alternatives_zh],['局限',d.caveats_zh]].forEach(([title,values])=>{const p=document.createElement('p');p.textContent=title+'：'+(values.join('；')||'—');box.append(p)});if(d.original_evidence){const a=document.createElement('a');a.href=d.original_evidence;a.textContent='查看原图与激活区域';a.target='_blank';a.rel='noopener';box.append(a)}const raw=document.createElement('a');raw.href=d.record;raw.textContent=' · 原始VLM回答与输入记录';box.append(raw);d.sheets.forEach(s=>{const img=document.createElement('img');img.src=s;img.loading='lazy';img.alt='VLM实际输入拼图';box.append(img)})}
document.querySelector('#search').oninput=listing;types.onchange=listing;window.addEventListener('hashchange',detail);listing();detail();</script></html>'''
    (args.report / 'index.html').write_text(page)
    body = ['# SAE 逐维 VLM 含义记录', '', f'已记录或明确无实例：{complete} / {WIDTH}。', '',
            '输入为已有 Top100 激活区域，不重提 DINO/SigLIP 特征，不重训 SAE。每维全部可用裁剪分成最多两张 50 格拼图，由本地 Qwen3-VL-2B-Instruct 共同解释。输入不含城市名称、64 组名称、embedding 一致性分数或激活颜色叠图。', '',
            '每维记录候选中文含义、类型、共同依据、异类/其他解释、一致性、模型自述 confidence 及局限。少于20例仅作小样本暂定解释，要求低信心；没有正激活不调用 VLM。', '',
            '**VLM 输出不是已经核实的语义标签。** 这份记录用于逐维核对；物体部件、纹理、颜色/光照、场景上下文和混合模式均被允许。模型可能漏掉少数异类，也可能给表面纹理误命名；可点击查看实际输入和原图。confidence 不是准确率，Top100 的解释不能外推到所有激活。', '',
            '原始回答、每次重试、输入图片、提示词和模型权重哈希均保留。程序检查 JSON 和字段，格式检查不能验证视觉内容正确。', '',
            '[交互逐维记录](index.html) · [CSV](dimension_interpretations.csv) · [JSON](dimension_interpretations.json)', '',
            '## 状态统计', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```']
    (args.report / 'REPORT.md').write_text('\n'.join(body) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--stage', choices=['run', 'prepare', 'worker', 'report'], default='run')
    parser.add_argument('--devices', default='1,3')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--features', default='')
    args = parser.parse_args()
    if args.stage in ['prepare', 'run']:
        prepare(args)
    if args.stage == 'worker':
        worker(args)
    elif args.stage == 'report':
        report(args)
    elif args.stage == 'run':
        devices = args.devices.split(',')
        processes = []
        for shard, device in enumerate(devices):
            cmd = [sys.executable, '-u', '-m', 'sae_experiments.panorama_vitl30_dimension_vlm',
                '--stage', 'worker', '--device', f'cuda:{device}', '--shard', str(shard), '--shards', str(len(devices)),
                '--batch-size', str(args.batch_size), '--max-new-tokens', str(args.max_new_tokens),
                '--source', str(args.source), '--report', str(args.report), '--model', str(args.model),
                '--features', args.features]
            log = (args.report / f'worker_{shard}.log').open('a', buffering=1)
            processes.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        while any(p.poll() is None for p, _ in processes):
            time.sleep(30)
            report(args)
        for p, log in processes:
            log.close()
        report(args)
        if any(p.returncode for p, _ in processes):
            raise SystemExit('A VLM worker failed; inspect worker logs; completed records are reusable')


if __name__ == '__main__':
    main()
