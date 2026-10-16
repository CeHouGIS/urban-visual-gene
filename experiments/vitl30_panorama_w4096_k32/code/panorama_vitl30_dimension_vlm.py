"""Describe SAE dimensions from ranked activation evidence, with explicit masking.

Default: Top10 distinct panoramas, high-activation patches only, native RGB
outside the binary mask replaced by neutral grey. Existing normalized 28x28
maps are reused; neither backbone features nor SAE training are repeated.
Descriptions are model hypotheses tied to these examples, not semantic truth.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageOps
import numpy as np

from formal.analysis.semantic_feature_umap import digest
from sae_experiments.panorama_vitl30 import OUTPUT, WIDTH
from sae_experiments.paths import REPO_ROOT

SOURCE = OUTPUT / 'analysis/semantic_umap_top100'
REPORT = REPO_ROOT / 'experiments/vitl30_panorama_w4096_k32/reports/vlm_dimensions_top10_masked'
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

MASK_INSTRUCTIONS = '''\n只识别保留下来的原图彩色像素：每个实例裁到高激活patch的包围框，框内未达到阈值的像素已置为统一灰色。灰色区域、矩形/阶梯边界和编号是人为mask及排版，不能作为语义依据。这个mask是28×28 patch归因，不是物体轮廓分割。比较最多Top10实例；若剩余区域太小或遮挡后无法识别，回答unclear，不要根据缺失背景猜测。'''


def evidence_settings(args):
    mode = getattr(args, 'input_mode', 'masked-top10')
    masked = mode == 'masked-top10'
    if mode not in {'masked-top10', 'clean-top100'}:
        raise ValueError('Unknown input mode')
    fraction = getattr(args, 'mask_fraction', .5)
    if not 0 < fraction <= 1:
        raise ValueError('Mask fraction must be in (0, 1]')
    return {'mode': mode, 'top': 10 if masked else 100,
            'tile_size': 224 if masked else 128, 'columns': 5 if masked else 10,
            'crops_per_sheet': 10 if masked else 50,
            'minimum_evidence': 5 if masked else 20,
            'mask_fraction': fraction if masked else None,
            'mask_background_rgb': [127, 127, 127] if masked else None,
            'mask_source': 'saved per-image peak-normalized uint8 28x28 activations' if masked else None,
            'mask_upsampling': 'nearest; no dilation or smoothing' if masked else None,
            'crop': 'tight union bounding box of all high-activation patches' if masked else 'existing peak-component crop',
            'prompt': PROMPT + MASK_INSTRUCTIONS if masked else PROMPT}


def mask_native(image, heatmap, fraction=.5, background=(127, 127, 127)):
    """Map a binary patch mask to native pixels, then crop and mask losslessly.

    Saved activations are uint8, round(255 * activation / per-view peak).
    Thresholding uses ceil(255 * fraction); all holes and separate regions are
    retained. No interpolation of activations or expansion of the support.
    """
    heatmap = np.asarray(heatmap)
    if heatmap.shape != (28, 28) or heatmap.dtype != np.uint8 or heatmap.max() != 255:
        raise ValueError('Expected peak-normalized uint8 28x28 activation map')
    if not 0 < fraction <= 1:
        raise ValueError('Mask fraction must be in (0, 1]')
    patch_mask = heatmap >= int(np.ceil(255 * fraction))
    native_mask = Image.fromarray(patch_mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    bbox = native_mask.getbbox()
    if bbox is None:
        raise ValueError('High-activation mask is empty at native resolution')
    clean = image.convert('RGB').crop(bbox)
    mask = native_mask.crop(bbox)
    masked = Image.composite(clean, Image.new('RGB', clean.size, background), mask)
    return clean, masked, mask, {'bbox_native_view': list(bbox),
        'kept_patches': int(patch_mask.sum()), 'threshold_uint8': int(np.ceil(255 * fraction)),
        'kept_pixel_fraction_of_crop': float(np.asarray(mask, dtype=bool).mean())}


def materialize_masked(source, rows, report_path, fraction):
    """Only opens requested Top10 images; no directory traversal or re-encoding."""
    result = []
    for row in rows:
        feature, rank = row['feature'], row['rank']
        folder = report_path / 'masked_crops' / f'f{feature:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f'r{rank:02d}'
        paths = {k: stem.with_name(stem.name + suffix) for k, suffix in
                 [('masked', '_masked.png'), ('mask', '_mask.png'), ('clean', '_clean.png'), ('audit', '.json')]}
        if not all(p.exists() for p in paths.values()):
            heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(row['heatmap_png']))))
            with Image.open(row['image']) as original:
                image = original.convert('RGB')  # same geometry as original activation extraction
            if list(image.size) != row['native_view_size']:
                raise ValueError('Original image geometry changed')
            clean, masked, mask, info = mask_native(image, heat, fraction)
            clean.save(paths['clean']); masked.save(paths['masked']); mask.save(paths['mask'])
            atomic_json(paths['audit'], {**info, 'feature': feature, 'rank': rank,
                'image': row['image'], 'activation_peak': row['activation'],
                'source_heatmap_sha256': hashlib.sha256(base64.b64decode(row['heatmap_png'])).hexdigest(),
                'mask_fraction': fraction, 'background_rgb': [127, 127, 127],
                'pixel_mask_sha256': digest(paths['mask']), 'masked_image_sha256': digest(paths['masked'])})
        result.append({**row, 'crop': os.path.relpath(paths['masked'], source),
                       'mask': os.path.relpath(paths['mask'], source)})
    return result


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
        if row.get('mask'):
            with Image.open(source / row['mask']) as binary:
                mask = binary.resize(crop.size, Image.Resampling.NEAREST)
            crop = Image.composite(crop, Image.new('RGB', crop.size, (127, 127, 127)), mask)
        panel.paste(crop, (x + (tile - crop.width) // 2, y + 16 + (tile - 20 - crop.height) // 2))
        draw.text((x + 4, y + 1), str(row['rank']), fill='#111111')
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.save(target, **({'quality': 90} if target.suffix.lower() in {'.jpg', '.jpeg'} else {}))
    return panel.size


def prepare(args):
    args.report.mkdir(parents=True, exist_ok=True)
    settings = evidence_settings(args)
    source_run = json.loads((args.source / 'run.json').read_text())
    manifest_hash = digest(args.source / 'crops.jsonl')
    weights = {p.name: digest(p) for p in sorted(args.model.glob('model*.safetensors'))}
    if not weights:
        raise ValueError('No local VLM weights')
    identity = {'version': 2, 'checkpoint_sha256': source_run['checkpoint_sha256'],
                'crop_manifest_sha256': manifest_hash, 'model': str(args.model.resolve()),
                'source_scope': {key: source_run[key] for key in ['cities', 'panoramas', 'images', 'scope', 'ranking']},
                'model_config_sha256': digest(args.model / 'config.json'), 'model_weights_sha256': weights,
                'prompt': settings['prompt'], 'evidence': settings,
                'tile_size': settings['tile_size'], 'columns': settings['columns'],
                'crops_per_sheet': settings['crops_per_sheet'],
                'max_new_tokens': args.max_new_tokens, 'precision': 'bfloat16',
                'generation': {'do_sample': False, 'repetition_penalty': 1.03},
                'code_sha256': digest(Path(__file__))}
    for key in ['sample_kind', 'sae_split_counts', 'activation_storage', 'target_features',
                'target_dimensions', 'selection', 'baseline_report', 'baseline_identity']:
        if key in source_run:
            identity['source_scope'][key] = source_run[key]
    requested = set(source_run.get('target_features', range(WIDTH)))
    if any(not isinstance(f, int) or not 0 <= f < WIDTH for f in requested):
        raise ValueError('Invalid target dimensions')
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = args.report / 'identity.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError('VLM run identity changed; choose a new report directory')
    atomic_json(target, identity)
    groups = defaultdict(list)
    for observation, line in enumerate((args.source / 'crops.jsonl').open()):
        row = json.loads(line)
        if int(row['feature']) not in requested:
            raise ValueError('Crop belongs to an already interpreted, excluded dimension')
        if int(row['rank']) <= settings['top']:
            groups[int(row['feature'])].append({**row, 'observation_index': observation,
                'feature': int(row['feature']), 'rank': int(row['rank']), 'pano_index': int(row['pano_index'])})
    evidence = []
    for feature in range(WIDTH):
        rows = sorted(groups[feature], key=lambda r: r['rank'])
        if [r['rank'] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f'Non-consecutive Top ranks for f{feature}')
        if any(r['activation'] <= 0 for r in rows):
            raise ValueError('Top ranks require positive activations')
        if len({r['pano_index'] for r in rows}) != len(rows):
            raise ValueError('Top100 must contain distinct panoramas')
        per_sheet = settings['crops_per_sheet']
        suffix = 'png' if settings['mode'] == 'masked-top10' else 'jpg'
        sheets = [f'inputs/f{feature:04d}_{i // per_sheet}.{suffix}' for i in range(0, len(rows), per_sheet)]
        evidence.append({'feature': feature, 'count': len(rows), 'crops': rows, 'sheets': sheets})
    atomic_json(args.report / 'evidence.json', {'identity': identity['identity'], 'dimensions': evidence})
    (args.report / 'records').mkdir(exist_ok=True)
    for row in evidence:
        if row['feature'] not in requested:
            atomic_json(args.report / 'records' / f'f{row["feature"]:04d}.json', {
                'feature': row['feature'], 'identity': identity['identity'], 'status': 'not_requested',
                'crop_count': 0, 'interpretation': None, 'independently_validated': False,
                'reason': 'Existing interpretation preserved in the baseline report; not reprocessed'})
        elif not row['count']:
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
    if not args.device.startswith('cuda:'):
        raise ValueError('VLM inference requires a CUDA GPU; CPU inference/offloading is disabled')
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    torch.manual_seed(42)
    identity = json.loads((args.report / 'identity.json').read_text())
    settings = evidence_settings(args)
    if settings != identity['evidence']:
        raise ValueError('Worker evidence settings differ from prepared run')
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
            for start, relative in zip(range(0, row['count'], settings['crops_per_sheet']), row['sheets']):
                path = args.report / relative
                if not path.exists():
                    sheet_rows = row['crops'][start:start + settings['crops_per_sheet']]
                    if settings['mode'] == 'masked-top10':
                        sheet_rows = materialize_masked(args.source, sheet_rows, args.report, settings['mask_fraction'])
                    contact_sheet(args.source, sheet_rows, path,
                                  tile=settings['tile_size'], columns=settings['columns'])
                with Image.open(path) as sheet:
                    content.append({'type': 'image', 'image': sheet.convert('RGB')})
            prompt = settings['prompt'] + f'\n本次共{row["count"]}个不同全景的裁剪。'
            if row['count'] < settings['minimum_evidence']:
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
                      'evidence_scope': f'Top{settings["top"]}: {identity["source_scope"]["scope"]}; {settings["mode"]}',
                      'input_mode': settings['mode']}
            outcome = {'attempt': attempt + 1, 'raw_response': raw, 'truncated': truncated,
                       'batch_seconds': elapsed, 'batch_size': len(batch)}
            try:
                interpretation = parse_response(raw, truncated)
                if row['count'] < settings['minimum_evidence'] and interpretation['confidence'] != 'low':
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
    settings = identity['evidence']
    scope = identity['source_scope']
    requested = set(scope.get('target_features', range(WIDTH)))
    expanded = scope.get('sample_kind') == 'expanded_posthoc'
    scope_description = (f'Top排名来自{len(scope["cities"])}城市、{scope["panoramas"]:,}个全景、'
        f'{scope["images"]:,}张方向图。' +
        ('这是包含SAE训练与验证图像的事后解释检索库，不是独立验证集。仅补做此前未解释的维度，已有解释保留。'
         if expanded else '这是固定验证样本，不代表全NAS。'))
    masked = settings['mode'] == 'masked-top10'
    explanation = (f'每维选既有排名前{settings["top"]}个不同全景的最强方向，复用28×28峰值归一化激活图；'
        f'保留达到该方向峰值{settings["mask_fraction"]:.0%}的patch，最近邻映射到原图像素，'
        '裁到所有保留区域的联合包围框，框内其余像素置灰（RGB 127）。PNG保存，空洞不填补、不平滑或膨胀。'
        '每维一张5列×2行拼图，单格224像素；Qwen只看masked原图像素，灰色不是语义。'
        if masked else '每维已有Top100干净激活区域，最多两张50格拼图，共同输入Qwen。')
    evidence = json.loads((args.report / 'evidence.json').read_text())['dimensions']
    audit_path = Path('/root/code/urban-visual-gene/reports/vitl30_unit_motifs/dimension_audit.json')
    audit = json.loads(audit_path.read_text()) if audit_path.exists() and not expanded else None
    rows, complete, data = [], 0, []
    for feature, item in enumerate(evidence):
        path = args.report / 'records' / f'f{feature:04d}.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        if record and record['identity'] != identity['identity']:
            raise ValueError('Stale record')
        annotation = record.get('interpretation') or {}
        status = record.get('status', 'pending')
        complete += feature in requested and status in {'interpreted', 'unobserved_in_sample'}
        original = args.source / 'features' / f'{feature}.html'
        row = {'feature': feature, 'status': status, 'crop_count': item['count'],
               'label_zh': annotation.get('label', ''), 'semantic_type': annotation.get('type', ''),
               'consistency': annotation.get('consistency', ''), 'confidence_self_reported': annotation.get('confidence', ''),
               'cues_zh': annotation.get('cues', []), 'alternatives_zh': annotation.get('alternatives', []),
               'caveats_zh': annotation.get('caveats', []), 'independently_validated': False,
               'embedding_evidence_status': audit[feature]['status'] if audit else '',
               'original_evidence': str(original.resolve()) if item['count'] and original.exists() else '',
               'record': f'records/f{feature:04d}.json'}
        rows.append(row)
        examples = []
        if masked:
            for crop in item['crops']:
                stem = f'masked_crops/f{feature:04d}/r{crop["rank"]:02d}'
                if (args.report / (stem + '.json')).exists():
                    examples.append({'rank': crop['rank'], 'activation': crop['activation'],
                                     'clean': stem + '_clean.png', 'masked': stem + '_masked.png',
                                     'mask': stem + '_mask.png', 'audit': stem + '.json'})
        data.append({**row, 'sheets': item['sheets'] if status == 'interpreted' else [], 'examples': examples})
    atomic_json(args.report / 'dimension_interpretations.json', {'identity': identity['identity'], 'dimensions': rows})
    target = args.report / 'dimension_interpretations.csv'
    with target.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    summary = {'dimensions': WIDTH, 'completed_or_explicitly_unobserved': complete,
               'requested_dimensions': len(requested),
               'status_counts': dict(Counter(r['status'] for r in rows)),
               'semantic_type_counts': dict(Counter(r['semantic_type'] for r in rows if r['status'] == 'interpreted')),
               'consistency_counts': dict(Counter(r['consistency'] for r in rows if r['status'] == 'interpreted')),
               'input_mode': settings['mode'], 'top_per_dimension': settings['top'],
               'input_instances': sum(r['crop_count'] for r in rows),
               'mask_fraction': settings['mask_fraction'], 'source_scope': identity['source_scope'],
               'model': str(args.model), 'independently_validated': False,
               'updated_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(args.report / 'summary.json', summary)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐维 VLM 含义记录</title><style>
body{font:15px system-ui;background:#f3f5f7;color:#243344;max-width:1250px;margin:auto;padding:24px}input,select,button{font:inherit;padding:9px;border:1px solid #acb8c4;border-radius:6px;background:white}header,section{background:white;padding:20px;border-radius:10px;margin-bottom:18px}.controls{display:flex;gap:8px;flex-wrap:wrap}p{line-height:1.7}a{color:#17698d}#listing{max-height:350px;overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #eee;text-align:left}tr[data-feature]{cursor:pointer}tr[data-feature]:hover{background:#e6f3ff}img{width:100%;height:auto}pre{white-space:pre-wrap}.muted{color:#627080}</style>
<header><h1>4,096 个 SAE 维度：VLM 含义记录</h1><p>__EVIDENCE_DESCRIPTION__</p><p>Qwen3-VL-2B 给出的名称是候选解释，confidence 是模型自述，均未经独立语义验证。无激活维度不编造含义。</p><p><a href="dimension_interpretations.csv">CSV</a> · <a href="dimension_interpretations.json">JSON</a> · <a href="REPORT.md">说明</a></p><p id="progress"></p></header><section><div class="controls"><input id="search" placeholder="搜索编号、名称或类型"><select id="type"><option value="">全部类型</option></select></div><div id="listing"></div></section><section id="detail"><h2>选择一个维度</h2></section>
'''
    page = page.replace('__EVIDENCE_DESCRIPTION__', html.escape(scope_description + explanation))
    page += '<script id="data" type="application/json">' + json.dumps(data, ensure_ascii=False).replace('<', '\\u003c') + '</script>'
    page += '<script>const data=JSON.parse(document.querySelector("#data").textContent);'
    page += f'document.querySelector("#progress").textContent="本次已记录或明确无实例：{complete} / {len(requested)}；未请求的维度沿用既有解释。";'
    page += '''const types=document.querySelector('#type');[...new Set(data.map(d=>d.semantic_type).filter(Boolean))].sort().forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;types.append(o)});
function listing(){const q=document.querySelector('#search').value.trim().toLowerCase();const subset=data.filter(d=>(!types.value||d.semantic_type===types.value)&&(!q||JSON.stringify([d.feature,d.label_zh,d.semantic_type]).toLowerCase().includes(q)));const table=document.createElement('table');table.innerHTML='<tr><th>维度</th><th>VLM候选含义</th><th>类型</th><th>实例数</th><th>状态</th></tr>';subset.forEach(d=>{const tr=document.createElement('tr');tr.dataset.feature=d.feature;[d.feature,d.label_zh||'—',d.semantic_type,d.crop_count,d.status].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});tr.onclick=()=>{location.hash='f'+d.feature};table.append(tr)});document.querySelector('#listing').replaceChildren(table)}
function detail(){const m=location.hash.match(/^#f(\\d+)$/);if(!m)return;const d=data[Number(m[1])];if(!d)return;const box=document.querySelector('#detail');box.replaceChildren();const h=document.createElement('h2');h.textContent=`f${d.feature} · ${d.label_zh||d.status}`;box.append(h);const p=document.createElement('p');p.textContent=`${d.crop_count} 个裁剪；${d.semantic_type}；一致性：${d.consistency}；模型自述信心：${d.confidence_self_reported}。`;box.append(p);[['共同依据',d.cues_zh],['异类/其他解释',d.alternatives_zh],['局限',d.caveats_zh]].forEach(([title,values])=>{const p=document.createElement('p');p.textContent=title+'：'+(values.join('；')||'—');box.append(p)});if(d.original_evidence){const a=document.createElement('a');a.href=d.original_evidence;a.textContent='查看原图与激活区域';a.target='_blank';a.rel='noopener';box.append(a)}const raw=document.createElement('a');raw.href=d.record;raw.textContent=' · 原始VLM回答与输入记录';box.append(raw);d.sheets.forEach(s=>{const img=document.createElement('img');img.src=s;img.loading='lazy';img.alt='VLM实际输入拼图';box.append(img)})}
const originalDetail=detail;detail=function(){originalDetail();const m=location.hash.match(/^#f(\\d+)$/);if(!m)return;const d=data[Number(m[1])];if(!d||!d.examples.length)return;const box=document.querySelector('#detail');const h=document.createElement('h3');h.textContent='逐实例核对：裁剪原图 → masked输入 → 二值mask';box.append(h);d.examples.forEach(e=>{const p=document.createElement('p');p.textContent='排名 '+e.rank+'；峰值 '+e.activation.toFixed(4);box.append(p);const row=document.createElement('div');row.style='display:flex;gap:8px;align-items:start';['clean','masked','mask'].forEach(k=>{const img=document.createElement('img');img.src=e[k];img.alt=k;img.loading='lazy';img.style='width:31%;height:auto';row.append(img)});box.append(row)})};
document.querySelector('#search').oninput=listing;types.onchange=listing;window.addEventListener('hashchange',detail);listing();detail();</script></html>'''
    (args.report / 'index.html').write_text(page)
    body = ['# SAE 逐维 VLM 含义记录', '', f'本次已记录或明确无实例：{complete} / {len(requested)}。', '',
            scope_description, '', explanation, '',
            '复用已有特征和激活，不重复训练。Qwen输入不含城市名称、64组名称、embedding一致性分数或激活颜色叠图。', '',
            f'每维记录候选中文含义、类型、共同依据、异类/其他解释、一致性、模型自述confidence及局限。少于{settings["minimum_evidence"]}例要求低信心；没有正激活不调用VLM。', '',
            '**VLM 输出不是已经核实的语义标签。** 允许物体部件、纹理、颜色/光照、场景上下文及混合模式。mask是patch级归因，不是像素语义分割；8bit激活图有量化误差，阈值128/255代表近似0.5峰值。模型可能受遮挡和patch边界影响；confidence不是准确率。', '',
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
    parser.add_argument('--input-mode', choices=['masked-top10', 'clean-top100'], default='masked-top10')
    parser.add_argument('--mask-fraction', type=float, default=.5)
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
                '--input-mode', args.input_mode, '--mask-fraction', str(args.mask_fraction),
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
