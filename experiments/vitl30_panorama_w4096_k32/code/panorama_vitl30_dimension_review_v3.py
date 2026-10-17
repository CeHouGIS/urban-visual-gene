"""Two-stage Qwen review of up to 40 activation examples per SAE dimension.

Four ten-example visual calls are followed by one text-only synthesis. A failed
or mixed answer is retained as evidence, never promoted to a verified label.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .panorama_vitl30_dimension_review import (
    EXPERIMENT, MODEL, SCREEN, SOURCE, atomic_json, sha,
)
from .panorama_vitl30_dimension_vlm import mask_native, parse_response

BASE = EXPERIMENT / 'reports/vlm_dimensions_top40_context_v2'
REPORT = EXPERIMENT / 'reports/vlm_dimensions_top40_masked_only_v6'
MODULE = 'experiments.vitl30_panorama_w4096_k32.code.panorama_vitl30_dimension_review_v3'
PART_PROMPT = '''这一张拼图展示同一个街景 SAE 维度在最多10个不同全景中的实例。每个编号中，左边是高激活区域，其他像素已变灰；右边是该区域未遮罩的裁剪。灰色和矩形边界是人工处理，不是景观。只根据左边保留下来的彩色区域判断，右边仅用于辨认该区域内的内容，不要根据其他背景推断整条街的类别。寻找反复出现的具体内容；位置、角度、形状与光照变化不等于语义混合。允许物体、部件、纹理、材质、强光、场景上下文。若确有多种互不相关的主要内容才用mixed；看不清写unclear。只输出JSON，恰好7个字段：label、type、cues、alternatives、consistency、confidence、caveats。type限object, object_part, vegetation, built_structure, surface_texture, color_light, scene_context, camera_artifact, mixed, unclear；consistency限mostly_consistent, mixed, unclear；confidence限low, medium, high。后三个列表各最多2条短句。label要是具体中文候选含义，不要写英文类型名；不能判断写“未确定”。'''
FINAL_PROMPT = '''下面四段JSON分别概括同一个SAE维度的四组不同图像（每组约10个全景）。现在比较四组观察，找跨组重复的真实视觉概念。把同一概念在颜色、形态、位置、角度、天气上的变化视为一致；不要因为原图背景不同就写mixed。若有两个以上互不相关的主要视觉概念，type=mixed；无法确认时type=unclear、label=“未确定”。灰色遮罩与裁剪边界是人工处理，不能成为标签或共同依据。不要编造四组没有观察到的细节。只输出JSON，恰好7个字段：label、type、cues、alternatives、consistency、confidence、caveats；type限object, object_part, vegetation, built_structure, surface_texture, color_light, scene_context, camera_artifact, mixed, unclear；consistency限mostly_consistent, mixed, unclear；confidence限low, medium, high。后三个列表各最多2条短句；label须为中文，不能写英文类型名。以下是四组观察：\n'''


def focused_sheet(rows, target):
    canvas = Image.new('RGB', (1600, 384), '#dedede')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        x, y = index % 5 * 320, index // 5 * 192
        draw.text((x + 3, y + 5), f'{row["rank"]}  active | crop', fill='#111111')
        with Image.open(row['image']) as opened:
            original = opened.convert('RGB')
        heat = Image.open(io.BytesIO(base64.b64decode(row['heatmap_png']))).convert('L')
        crop, masked, _, _ = mask_native(original, np.asarray(heat, dtype=np.uint8), .5)
        for part, width, at in [(masked, 200, x), (crop, 120, x + 200)]:
            image = ImageOps.contain(part, (width, 160), Image.Resampling.BICUBIC)
            canvas.paste(image, (at + (width - image.width) // 2, y + 32 + (160 - image.height) // 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, 'JPEG', quality=90, subsampling=0)


def prepare(args):
    base = json.loads((args.base / 'identity.json').read_text())
    screen = json.loads(args.screen.read_text())
    if base['protocol']['screen_identity'] != screen['identity']:
        raise ValueError('Base evidence and screening identity differ')
    evidence = args.base / 'evidence.jsonl'
    protocol = {'version': 1, 'base_evidence_identity': base['identity'],
                'evidence_sha256': sha(evidence), 'screen_identity': screen['identity'],
                'checkpoint_sha256': base['protocol']['checkpoint_sha256'],
                'model_weights_sha256': base['protocol']['model_weights_sha256'],
                'part_prompt': PART_PROMPT, 'final_prompt': FINAL_PROMPT,
                'code_sha256': sha(Path(__file__)),
                'selection': 'four disjoint rank bands, up to ten panoramas per visual call',
                'semantic_validation': 'pending'}
    identity = sha_identity(protocol)
    args.report.mkdir(parents=True, exist_ok=True)
    path = args.report / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != {'identity': identity, 'protocol': protocol}:
        raise ValueError('Review identity changed; select a new report directory')
    atomic_json(path, {'identity': identity, 'protocol': protocol})
    print(json.dumps({'identity': identity, 'evidence': str(evidence)}, ensure_ascii=False), flush=True)


def sha_identity(protocol):
    import hashlib
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def label_quality(value):
    if value is None:
        return 'invalid'
    if value['type'] in {'mixed', 'unclear'} or value['consistency'] != 'mostly_consistent':
        return 'mixed_or_unclear'
    if value['label'].strip().lower() in {'mixed', 'unclear', 'vegetation', 'object', 'scene_context'}:
        return 'non_specific_label'
    if value['confidence'] == 'low':
        return 'low_confidence'
    return 'candidate_needs_validation'


def review_quality(parts, final):
    if len(parts) < 4 or any(part.get('interpretation') is None for part in parts):
        return 'incomplete_visual_evidence'
    if any(part['interpretation']['consistency'] != 'mostly_consistent' for part in parts):
        return 'mixed_or_unclear'
    types = {part['interpretation']['type'] for part in parts}
    if len(types) != 1 or final is None or final['type'] not in types:
        return 'cross_group_type_conflict'
    return label_quality(final)


def worker(args):
    if not args.device.startswith('cuda:'):
        raise ValueError('Visual review requires CUDA')
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    run = json.loads((args.report / 'identity.json').read_text())
    if run['protocol']['code_sha256'] != sha(Path(__file__)):
        raise ValueError('Review code changed after preparation')
    wanted = set(map(int, args.features.split(','))) if args.features else None
    rows = [json.loads(line) for line in (args.base / 'evidence.jsonl').open()]
    rows = [r for r in rows if r['feature'] % args.shards == args.shard
            and (wanted is None or r['feature'] in wanted)]
    pending = []
    for row in rows:
        path = args.report / 'records' / f'f{row["feature"]:04d}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('identity') == run['identity'] and previous.get('status') == 'completed':
                continue
        pending.append(row)
    if not pending:
        print('NOTHING PENDING', args.shard, flush=True)
        return
    device = torch.device(args.device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    started = time.monotonic()

    def infer(prompt, image=None):
        content = [] if image is None else [{'type': 'image', 'image': image}]
        content.append({'type': 'text', 'text': prompt})
        inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
            processor_kwargs={'images_kwargs': {'do_resize': False}}).to(device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                repetition_penalty=1.03)
        return processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)[0]

    def parsed(prompt, image=None):
        attempts = []
        for attempt in range(2):
            extra = ('' if not attempt else '\n上次回答的JSON格式不合格；请补齐全部7字段，三个列表各最多2项，不要Markdown。')
            raw = infer(prompt + extra, image)
            try:
                value = parse_response(raw)
                attempts.append({'raw': raw, 'error': None})
                return value, attempts
            except (ValueError, TypeError, KeyError) as exc:
                attempts.append({'raw': raw, 'error': str(exc)})
        return None, attempts

    for index, row in enumerate(pending, 1):
        feature = row['feature']
        path = args.report / 'records' / f'f{feature:04d}.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        parts = list(previous.get('parts', []))
        for group in range(len(parts), (len(row['examples']) + 9) // 10):
            target = args.report / 'inputs' / f'f{feature:04d}_{group}.jpg'
            if not target.exists():
                focused_sheet(row['examples'][group * 10:(group + 1) * 10], target)
            with Image.open(target) as opened:
                image = opened.convert('RGB')
            value, attempts = parsed(PART_PROMPT, image)
            parts.append({'group': group, 'count': len(row['examples'][group * 10:(group + 1) * 10]),
                          'sheet': str(target.relative_to(args.report)),
                          'interpretation': value, 'attempts': attempts})
            atomic_json(path, {'feature': feature, 'identity': run['identity'],
                'status': 'partial', 'parts': parts, 'instance_count': len(row['examples'])})
        observations = [part['interpretation'] if part['interpretation'] is not None
                        else {'unparsed_model_output': part['attempts'][-1]['raw']}
                        for part in parts]
        final_prompt = FINAL_PROMPT + json.dumps(observations, ensure_ascii=False)
        final, attempts = parsed(final_prompt)
        quality = review_quality(parts, final)
        atomic_json(path, {'feature': feature, 'identity': run['identity'],
            'status': 'completed', 'quality': quality,
            'instance_count': len(row['examples']),
            'cities': len({r['city'] for r in row['examples']}), 'parts': parts,
            'interpretation': final, 'final_attempts': attempts,
            'independently_validated': False,
            'updated_utc': datetime.now(timezone.utc).isoformat()})
        print('REVIEW_V3', args.shard, index, '/', len(pending), 'f'+str(feature),
              quality, 'elapsed', round(time.monotonic() - started), flush=True)


def report(args):
    run = json.loads((args.report / 'identity.json').read_text())
    rows = [json.loads(path.read_text()) for path in (args.report / 'records').glob('f*.json')]
    rows = [r for r in rows if r.get('identity') == run['identity']]
    summary = {'identity': run['identity'], 'target_dimensions': sum(1 for _ in (args.base / 'evidence.jsonl').open()),
        'completed': sum(r['status'] == 'completed' for r in rows),
        'partial': sum(r['status'] == 'partial' for r in rows),
        'quality_counts': dict(Counter(r.get('quality') for r in rows if r['status'] == 'completed')),
        'semantic_validation': 'pending'}
    atomic_json(args.report / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
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
                '--source', str(args.source), '--base', str(args.base),
                '--screen', str(args.screen), '--report', str(args.report),
                '--model', str(args.model), '--features', args.features]
            log = (args.report / f'worker_{shard}.log').open('a', buffering=1)
            processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        while any(process.poll() is None for process, _ in processes):
            time.sleep(30)
            report(args)
        for process, log in processes:
            log.close()
        report(args)
        if any(process.returncode for process, _ in processes):
            raise SystemExit('Review worker failed; inspect worker logs')


if __name__ == '__main__':
    main()
