"""Post-hoc Qwen descriptions of native image instances of candidate prototypes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from PIL import Image

from sae_experiments.io_utils import digest
from sae_experiments.panorama_vitl30_dimension_vlm import MODEL, parse_response
from .data import OUT, write_json

PROMPT = '''你在解释一个自动学习的候选景观原型。拼图中每格是不同街景全景的局部裁剪，同一卷积模板在这些位置响应较高。比较全部例子，描述共同可见的景观组成，以及它们重复出现的相对布局。数字只是排名，灰色只是排版；不要依据它们解释语义。不要推断城市、地理、不可读文字或不可见对象。
不要假设原型必然有明确语义；若只对应一种物体、纹理、光照或相机伪影，就如实说明；若例子混杂或缺乏稳定组合，说明mixed或unclear。只有多个可见元素及相对关系反复出现时，才称为候选景观组合。裁剪边界是模板归因范围，不是物体分割。
只输出JSON，必须包含全部7个字段：label、type、cues、alternatives、consistency、confidence、caveats。
type从object, object_part, vegetation, built_structure, surface_texture, color_light, scene_context, camera_artifact, mixed, unclear中选。
consistency从mostly_consistent, mixed, unclear中选；confidence从low, medium, high中选，是主观信心而非准确率。
label是具体中文候选名称。cues写共同可见元素和布局；alternatives写实际异类或其他解释；caveats写局限。三个列表各最多2条短句，总计不超过160中文字。不输出Markdown。'''


def interpret(out=OUT, device='cuda:3'):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = Path(out)
    prototypes = json.loads((out / 'gallery/prototypes.json').read_text())
    folder = out / 'interpretation'
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'gallery_sha256': digest(out / 'gallery/prototypes.json'),
        'prompt': PROMPT, 'model': str(MODEL), 'model_config_sha256': digest(MODEL / 'config.json'),
        'weights_sha256': {p.name: digest(p) for p in sorted(MODEL.glob('model*.safetensors'))},
        'code_sha256': digest(Path(__file__)), 'generation': {'sample': False, 'max_new_tokens': 448,
            'dtype': 'bfloat16', 'evidence': 'up to eight unmasked native crops, distinct panoramas; city diversity preferred'}}
    if not identity['weights_sha256']:
        raise ValueError('Missing local Qwen weights')
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    old = folder / 'identity.json'
    if old.exists() and json.loads(old.read_text()) != identity:
        raise ValueError('Prototype VLM identity changed')
    write_json(old, identity)
    records, pending = [], []
    for p in prototypes:
        path = folder / f'p{p["prototype"]:03d}.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['identity'] != identity['identity']:
                raise ValueError('Stale prototype interpretation')
            if previous['status'] == 'interpreted':
                parse_response(previous['raw_response'], previous.get('truncated', False))
                records.append(previous)
                continue
        if not p['instances']:
            record = {'prototype': p['prototype'], 'identity': identity['identity'],
                      'status': 'no_positive_evidence', 'interpretation': None,
                      'independently_validated': False}
            write_json(path, record)
            records.append(record)
        else:
            pending.append(p)
    if pending:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        failures = []
        for p in pending:
            prototype = p['prototype']
            path = folder / f'p{prototype:03d}.json'
            sheet = out / 'gallery' / f'p{prototype:03d}' / 'vlm_sheet.jpg'
            with Image.open(sheet) as opened:
                image = opened.convert('RGB')
            attempts = []
            for attempt in range(3):
                prompt = PROMPT + f'\n本组有{len(p["instances"])}个不同全景的例子。'
                if len(p['instances']) < 5:
                    prompt += '证据少于5例，confidence必须为low，并在caveats说明证据不足。'
                if attempt:
                    prompt += '上次JSON格式无效。请检查全部7个字段及枚举值，不要漏掉type。'
                messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                         {'type': 'text', 'text': prompt}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors='pt',
                    processor_kwargs={'images_kwargs': {'do_resize': False}}).to(device)
                limit = 448 + attempt * 64
                with torch.inference_mode():
                    tokens = model.generate(**inputs, max_new_tokens=limit, do_sample=False, repetition_penalty=1.03)
                generated = tokens[0, inputs['input_ids'].shape[1]:]
                raw = processor.decode(generated, skip_special_tokens=True)
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos)
                truncated = len(generated) >= limit and not any(int(t) in eos for t in generated)
                record = {'prototype': prototype, 'identity': identity['identity'], 'raw_response': raw,
                    'truncated': truncated, 'instances': len(p['instances']), 'sheet_sha256': digest(sheet),
                    'updated_utc': datetime.now(timezone.utc).isoformat(), 'independently_validated': False}
                try:
                    parsed = parse_response(raw, truncated)
                    if len(p['instances']) < 5 and parsed['confidence'] != 'low':
                        raise ValueError('Small evidence set requires low confidence')
                    record.update(status='interpreted', interpretation=parsed)
                    attempts.append({'attempt': attempt + 1, 'raw_response': raw, 'valid': True})
                except (ValueError, TypeError) as exc:
                    record.update(status='response_invalid', interpretation=None)
                    attempts.append({'attempt': attempt + 1, 'raw_response': raw, 'valid': False, 'error': str(exc)})
                record['attempts'] = list(attempts)
                write_json(path, record)
                del inputs, tokens, generated
                if record['status'] == 'interpreted':
                    break
            records.append(record)
            if record['status'] != 'interpreted':
                failures.append(prototype)
            write_json(folder / 'status.json', {'processed': len(records), 'total': len(prototypes),
                                                'invalid_responses': failures})
            print('PROTOTYPE_VLM', prototype, record['status'], flush=True)
        del model, processor
        torch.cuda.empty_cache()
    else:
        failures = []
    records.sort(key=lambda r: r['prototype'])
    write_json(folder / 'records.json', records)
    sections = []
    for r in records:
        parsed = r.get('interpretation') or {}
        label = html.escape(parsed.get('label', r['status']))
        info = html.escape(json.dumps(parsed, ensure_ascii=False, indent=2))
        p = r['prototype']
        sections.append(f'<section><h2>P{p:03d} · {label}</h2><a href="../gallery/index.html#p{p}">核查原图及热图</a><pre>{info}</pre><img loading="lazy" src="../gallery/p{p:03d}/vlm_sheet.jpg"></section>' if r['status'] != 'no_positive_evidence' else f'<section><h2>P{p:03d} · 无正响应证据</h2></section>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>候选景观原型解释</title><style>body{max-width:1100px;margin:auto;font:16px system-ui}section{margin:30px 0;border-top:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Qwen 对候选景观原型的事后解释</h1><p>名称和布局描述是模型假设，需要与原图、对照和稳定性共同复核。</p>' + ''.join(sections))
    if failures:
        raise RuntimeError(f'Invalid prototype VLM responses after retries: {failures}; rerun interpretation to resume')
    write_json(folder / 'complete.json', {'identity': identity['identity'],
        'interpreted': sum(r['status'] == 'interpreted' for r in records),
        'no_positive_evidence': sum(r['status'] == 'no_positive_evidence' for r in records),
        'independently_validated': False})
