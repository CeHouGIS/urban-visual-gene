"""Recheck supported SAE dimensions with diverse Top40 image evidence.

This is post-hoc interpretation, not independent validation. Each Qwen input
shows the original direction, its SAE activation overlay and the clean crop.
Existing DINO features, SAE checkpoint and Top100 ranking are reused.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .panorama_vitl30_dimension_vlm import MODEL, parse_response

EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENT / 'outputs/analysis/semantic_umap_top100'
SCREEN = EXPERIMENT / 'results/dimension_screening/screening.json'
REPORT = EXPERIMENT / 'reports/vlm_dimensions_top40_context_v2'
MODULE = 'experiments.vitl30_panorama_w4096_k32.code.panorama_vitl30_dimension_review'
PROMPT = '''你在复核同一个街景 SAE 维度。每个编号实例有三幅图：完整方向原图、黄色 SAE 激活叠加图、高激活区域的干净裁剪。共有最多40个不同全景，按4张拼图呈现。比较不同城市和不同激活强度的全部实例，先寻找可见内容的共同点，再给维度一个暂定名称。只根据黄色激活所在的视觉内容判断，不把黄色颜色本身、裁剪边界、编号或同一图的背景当成特征。不要强迫每个维度对应物体；也可能是部件、材质、纹理、光照、场景上下文或混合响应。若实例含义不一致，type=mixed；缺乏足够清楚的共同内容，type=unclear。图像上相似的位置不能证明位置编码；只可在caveats中提出待检验的可能性。不要推断城市、品牌或不可读文字。
只输出一个JSON对象，必须恰好包含label、type、cues、alternatives、consistency、confidence、caveats七个字段。type从object, object_part, vegetation, built_structure, surface_texture, color_light, scene_context, camera_artifact, mixed, unclear中选；consistency从mostly_consistent, mixed, unclear中选；confidence从low, medium, high中选。cues、alternatives、caveats各最多2条短句。confidence是主观判断，不是准确率。若无法命名，label写“未确定”，confidence写low。'''


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def select_examples(rows, limit=40):
    """Cover strong and moderate responses while spreading each rank band by city."""
    bands = ((1, 10), (11, 25), (26, 50), (51, 100))
    ordered = sorted((r for r in rows if r['activation'] > 0), key=lambda r: r['rank'])
    if len({r['pano_index'] for r in ordered}) != len(ordered):
        raise ValueError('Top100 ranking must contain distinct panoramas')
    picked, used = [], set()
    for low, high in bands:
        pool = [r for r in ordered if low <= r['rank'] <= high]
        cities = Counter()
        while pool and len([r for r in picked if low <= r['rank'] <= high]) < limit // 4:
            row = min(pool, key=lambda r: (cities[r['city']], r['rank']))
            pool.remove(row)
            picked.append(row)
            used.add(row['rank'])
            cities[row['city']] += 1
    for row in ordered:
        if len(picked) >= limit:
            break
        if row['rank'] not in used:
            picked.append(row)
    return sorted(picked, key=lambda r: r['rank'])


def prepare(args):
    screen = json.loads(args.screen.read_text())
    source_run = json.loads((args.source / 'run.json').read_text())
    if screen['protocol']['checkpoint_sha256'] != source_run['checkpoint_sha256']:
        raise ValueError('Screen and activation ranking use different checkpoints')
    weights = {p.name: sha(p)
               for p in sorted(args.model.glob('model*.safetensors'))}
    if not weights:
        raise ValueError('Qwen weights not available locally')
    protocol = {'version': 1, 'screen_identity': screen['identity'],
                'checkpoint_sha256': source_run['checkpoint_sha256'],
                'source_run_sha256': sha(args.source / 'run.json'),
                'source_crops_sha256': sha(args.source / 'crops.jsonl'),
                'code_sha256': sha(Path(__file__)),
                'model_weights_sha256': weights, 'prompt': PROMPT, 'limit': 40,
                'selection': 'up to 10 panoramas per activation-rank band, diversified by city',
                'evidence': 'original view + SAE overlay + clean crop for each selected panorama',
                'scope': 'fixed validation sample; not an independent semantic test'}
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.report.mkdir(parents=True, exist_ok=True)
    path = args.report / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != {'identity': identity, 'protocol': protocol}:
        raise ValueError('Review identity changed; use a different report directory')
    atomic_json(path, {'identity': identity, 'protocol': protocol})
    eligible = {r['feature'] for r in screen['dimensions'] if r['label_eligible'] and r['support_scope'] == 'fixed_validation'}
    groups = defaultdict(list)
    with (args.source / 'crops.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['feature'] in eligible:
                groups[row['feature']].append(row)
    manifest = args.report / 'evidence.jsonl'
    if not manifest.exists():
        with manifest.open('w') as stream:
            for feature in sorted(eligible):
                chosen = select_examples(groups[feature])
                if len(chosen) < 20:
                    raise ValueError(f'f{feature} passed support screen but has fewer than 20 crops')
                stream.write(json.dumps({'feature': feature, 'examples': chosen}, ensure_ascii=False) + '\n')
    print(json.dumps({'prepared_dimensions': len(eligible), 'identity': identity}, ensure_ascii=False), flush=True)


def panel_image(row, source, size=128):
    with Image.open(row['image']) as image:
        original = ImageOps.contain(image.convert('RGB'), (size, size), Image.Resampling.BICUBIC)
    full = Image.new('RGB', (size, size), '#e5e5e5')
    full.paste(original, ((size - original.width) // 2, (size - original.height) // 2))
    heat = Image.open(io.BytesIO(base64.b64decode(row['heatmap_png']))).convert('L')
    if heat.size != (28, 28):
        raise ValueError('Expected 28x28 patch activation map')
    mask = np.asarray(heat.resize((size, size), Image.Resampling.NEAREST), dtype=np.float32) / 255
    alpha = np.where(mask >= .5, mask * .72, 0)[..., None]
    overlay = Image.fromarray(np.uint8(np.clip(
        np.asarray(full, dtype=np.float32) * (1 - alpha) +
        np.array([255, 207, 27], np.float32) * alpha, 0, 255)))
    with Image.open(source / row['crop']) as image:
        crop = ImageOps.contain(image.convert('RGB'), (size, size), Image.Resampling.BICUBIC)
    clean = Image.new('RGB', (size, size), '#e5e5e5')
    clean.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return full, overlay, clean


def sheet(rows, source, target):
    tile, gap, heading = 128, 0, 32
    stride = tile * 3 + gap
    canvas = Image.new('RGB', (stride * 5, (tile + heading) * 2), '#e5e5e5')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = i % 5 * stride, i // 5 * (tile + heading)
        draw.text((x + 2, y + 2), f'{row["rank"]}  {row["city"].split("/")[-1]}', fill='#111111')
        for j, part in enumerate(panel_image(row, source, tile)):
            canvas.paste(part, (x + j * tile, y + heading))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, 'JPEG', quality=88, subsampling=0)


def worker(args):
    if not args.device.startswith('cuda:'):
        raise ValueError('Qwen review requires a CUDA GPU')
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    review = json.loads((args.report / 'identity.json').read_text())
    identity = review['identity']
    if review['protocol']['code_sha256'] != sha(Path(__file__)):
        raise ValueError('Review code changed after evidence preparation')
    wanted = set(map(int, args.features.split(','))) if args.features else None
    rows = [json.loads(line) for line in (args.report / 'evidence.jsonl').open()]
    rows = [r for r in rows if r['feature'] % args.shards == args.shard
            and (wanted is None or r['feature'] in wanted)]
    pending = [r for r in rows if not (args.report / 'records' / f'f{r["feature"]:04d}.json').exists()]
    if not pending:
        print('NOTHING PENDING', args.shard, flush=True)
        return
    device = torch.device(args.device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    started = time.monotonic()
    for index, row in enumerate(pending, 1):
        feature = row['feature']
        paths = [args.report / 'inputs' / f'f{feature:04d}_{i}.jpg'
                 for i in range((len(row['examples']) + 9) // 10)]
        for i, path in enumerate(paths):
            if not path.exists():
                sheet(row['examples'][i * 10:(i + 1) * 10], args.source, path)
        content = []
        for path in paths:
            with Image.open(path) as image:
                content.append({'type': 'image', 'image': image.convert('RGB')})
        content.append({'type': 'text', 'text': PROMPT + f'\n实际实例数：{len(row["examples"])}。'})
        inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
            processor_kwargs={'images_kwargs': {'do_resize': False}}).to(device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                    repetition_penalty=1.03)
        raw = processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0]
        try:
            interpretation = parse_response(raw)
            status, error = 'interpreted', None
        except (ValueError, TypeError, KeyError) as exc:
            interpretation, status, error = None, 'response_invalid', str(exc)
        atomic_json(args.report / 'records' / f'f{feature:04d}.json', {
            'feature': feature, 'identity': identity, 'status': status, 'error': error,
            'instance_count': len(row['examples']), 'cities': len({r['city'] for r in row['examples']}),
            'sheets': [str(p.relative_to(args.report)) for p in paths],
            'raw_response': raw, 'interpretation': interpretation,
            'independently_validated': False,
            'updated_utc': datetime.now(timezone.utc).isoformat()})
        print('REVIEW', args.shard, index, '/', len(pending), 'f'+str(feature), status,
              'elapsed', round(time.monotonic() - started), flush=True)
        del inputs, tokens


def report(args):
    screen = json.loads(args.screen.read_text())
    identity = json.loads((args.report / 'identity.json').read_text())['identity']
    results = {}
    for path in (args.report / 'records').glob('f*.json'):
        row = json.loads(path.read_text())
        if row['identity'] == identity:
            results[row['feature']] = row
    eligible = [r for r in screen['dimensions'] if r['label_eligible'] and r['support_scope'] == 'fixed_validation']
    summary = {'identity': identity, 'target_dimensions': len(eligible),
               'completed': len(results), 'status_counts': dict(Counter(r['status'] for r in results.values())),
               'evidence_scope': 'fixed validation Top100; images include original, SAE overlay and clean crop',
               'semantic_validation': 'pending'}
    atomic_json(args.report / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--screen', type=Path, default=SCREEN)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--stage', choices=['prepare', 'worker', 'report', 'run'], default='prepare')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--devices', default='0,1,2,3')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--features', default='')
    args = parser.parse_args()
    if args.stage in {'prepare', 'run'}:
        prepare(args)
    if args.stage == 'worker':
        worker(args)
    elif args.stage == 'report':
        report(args)
    elif args.stage == 'run':
        devices = args.devices.split(',')
        processes = []
        for shard, device in enumerate(devices):
            command = [sys.executable, '-u', '-m', MODULE, '--stage', 'worker',
                '--device', f'cuda:{device}', '--shard', str(shard), '--shards', str(len(devices)),
                '--source', str(args.source), '--screen', str(args.screen),
                '--report', str(args.report), '--model', str(args.model), '--features', args.features]
            log = (args.report / f'worker_{shard}.log').open('a', buffering=1)
            processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        while any(p.poll() is None for p, _ in processes):
            time.sleep(30)
            report(args)
        for process, log in processes:
            log.close()
        report(args)
        if any(process.returncode for process, _ in processes):
            raise SystemExit('Review worker failed; inspect worker logs')


if __name__ == '__main__':
    main()
