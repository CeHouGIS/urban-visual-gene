"""Build a reversible review queue for spatially locked and blurry SAE dimensions."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/vitl30_panorama_w4096_k32'
HIERARCHY = ROOT / 'experiments/vitl30_relational_prototypes/results/hierarchy.json'
SOURCE = EXPERIMENT / 'outputs/analysis/semantic_umap_top100'
SCREENING = EXPERIMENT / 'results/dimension_screening/screening.json'
TEMPLATES = Path(__file__).with_name('templates')
BLUR_WORDS = re.compile(r'模糊|失焦|虚焦|拖影|运动模糊', re.I)
VISUAL_BLUR_REVIEW = {694, 900, 3598, 4077}
POSITION_FAMILIES = {'sky', 'road', 'ground_unspecified'}
REVIEW_NOTES = {
    1711: '裁剪横跨不同物体和纹理；“模糊”可能只是低细节或低分辨率。',
    3829: '多张高激活裁剪不清楚，完整方向原图并非整体失焦。',
    694: '多例为近景人脸；需区分人脸语义与匿名化模糊。',
    900: '多例为低清晰度人脸；需区分人脸语义与匿名化模糊。',
    3598: '响应包含头发和人脸近景；低清晰度未必是主要触发条件。',
    4077: '响应包含人物和面部；部分实例呈匿名化或低清晰度。',
}


def priority_position_review(screen, family):
    """Prioritize strong spatial concentration without declaring position coding."""
    return (screen['position_review'] and screen['peak_position_examples'] >= 100
            and screen['peak_4x4_max_bin_fraction'] >= .98
            and screen['positive_panoramas'] >= 2000
            and screen['top_crop_cities'] >= 20 and family in POSITION_FAMILIES)


def blur_evidence(label, fine_label, feature):
    if BLUR_WORDS.search((label or '') + ' ' + (fine_label or '')):
        return 'old_qwen_label'
    if feature in VISUAL_BLUR_REVIEW:
        return 'visual_spot_check'
    return ''


def peak_bins(path):
    bins = defaultdict(lambda: [0] * 16)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            y, x = divmod(int(row['peak_patch']) % 784, 28)
            bins[int(row['feature'])][(y // 7) * 4 + x // 7] += 1
    return bins


def collect(site):
    screening = json.loads(SCREENING.read_text())
    hierarchy = json.loads(HIERARCHY.read_text())
    interpretations = json.loads((site / 'interpretation/dimensions.json').read_text())
    published_hierarchy = json.loads((site / 'elements/hierarchy.json').read_text())
    assert screening['protocol']['checkpoint_sha256'] == hierarchy['protocol']['checkpoint_sha256']
    assert interpretations['dimension_screening_identity'] == screening['identity']
    assert published_hierarchy['dimension_screening_identity'] == screening['identity']
    assert published_hierarchy['identity'] == hierarchy['identity']
    ranking_sha256 = hashlib.sha256((SOURCE / 'top_images.jsonl').read_bytes()).hexdigest()
    ranking_source = str((SOURCE / 'top_images.jsonl').relative_to(ROOT))
    assert ranking_sha256 == screening['protocol']['inputs_sha256'][ranking_source]
    screens, labels, evidence = screening['dimensions'], hierarchy['dimensions'], interpretations['dimensions']
    assert len(screens) == len(labels) == len(evidence) == 4096
    assert all(s['feature'] == h['feature'] == e['feature'] == i
               for i, (s, h, e) in enumerate(zip(screens, labels, evidence)))
    bins = peak_bins(SOURCE / 'top_images.jsonl')
    rows = []
    for feature, (screen, label, interpretation) in enumerate(zip(screens, labels, evidence)):
        position = 'priority_review' if priority_position_review(screen, label['family']) else (
            'review' if screen['position_review'] else '')
        blur = blur_evidence(label['label_zh'], label['fine_label_zh'], feature)
        if not position and not blur:
            continue
        sheet = interpretation.get('sheet', '')
        sheet_url = '../interpretation/' + sheet if sheet else ''
        if sheet_url and not (site / 'interpretation' / sheet).exists():
            raise FileNotFoundError(f'Missing masked Qwen evidence for f{feature}: {sheet}')
        rows.append({
            'feature': feature, 'position_status': position, 'blur_evidence': blur,
            'label_zh': label['label_zh'], 'label_is_provisional': True,
            'support_status': screen['status'], 'support_scope': screen['support_scope'],
            'positive_panoramas': screen['positive_panoramas'],
            'top_crop_cities': screen['top_crop_cities'],
            'peak_position_examples': screen['peak_position_examples'],
            'peak_4x4_max_bin_fraction': screen['peak_4x4_max_bin_fraction'],
            'peak_4x4_bins': bins[feature],
            'masked_sheet': sheet_url,
            'review_note': REVIEW_NOTES.get(feature, ''),
        })
    rows.sort(key=lambda r: (r['position_status'] != 'priority_review',
                             r['position_status'] != 'review',
                             r['blur_evidence'] == '',
                             -r['positive_panoramas'], r['feature']))
    return screening, hierarchy, rows


def install(site, render_page, snapshot):
    screening, hierarchy, rows = collect(site)
    target = site / 'quality'
    target.mkdir(exist_ok=True)
    counts = {
        'priority_position_review': sum(r['position_status'] == 'priority_review' for r in rows),
        'position_review': sum(bool(r['position_status']) for r in rows),
        'blur_review': sum(bool(r['blur_evidence']) for r in rows),
        'old_qwen_blur_labels': sum(r['blur_evidence'] == 'old_qwen_label' for r in rows),
    }
    payload = {'checkpoint_sha256': screening['protocol']['checkpoint_sha256'],
               'screening_identity': screening['identity'],
               'hierarchy_identity': hierarchy['identity'],
               'ranking_sha256': screening['protocol']['inputs_sha256'][str((SOURCE / 'top_images.jsonl').relative_to(ROOT))],
               'position_rule': 'Priority review when Top100 peaks: >=98% in one 7x7 bin; >=2000 positive validation panoramas; >=20 cities; provisional sky/road/ground family',
               'blur_rule': 'old Qwen label keyword or limited visual spot check; neither is a verified blur detector',
               'scope': 'review queue only; no dimension has been dropped or excluded from the SAE checkpoint or existing analyses',
               'counts': counts, 'dimensions': rows}
    (target / 'candidates.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    with (target / 'candidates.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['feature', 'position_status', 'blur_evidence', 'label_zh', 'support_status',
                  'positive_panoramas', 'top_crop_cities', 'peak_position_examples',
                  'peak_4x4_max_bin_fraction', 'review_note']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    content = f'''<div class="eyebrow">SAE dimension quality review</div><h1>维度质量审阅</h1>
<p class="lead">先审阅固定位置响应和疑似模糊响应，再决定从景观单元分析中去掉哪些维度。</p>
<p class="note">{counts['priority_position_review']} 维优先复核，另有 {counts['position_review'] - counts['priority_position_review']} 维位置集中待查；{counts['blur_review']} 维列入模糊审阅。这里的“位置”仅指 Top100 峰值集中，天空与道路的真实位置也会造成这一现象。“模糊”来自旧 Qwen 名称或少量目视检查，并非已验证标签。现在没有删除或屏蔽任何维度。</p>
<div class="quality-controls"><div class="quality-tabs" aria-label="审阅类型"><button id="position-tab" type="button" aria-pressed="true">位置响应</button><button id="blur-tab" type="button" aria-pressed="false">模糊响应</button></div><input id="quality-search" aria-label="搜索维度" placeholder="搜索维度编号或旧标签"><button id="export-drop" type="button">导出待 drop 清单</button></div>
<p id="quality-count" class="muted"></p><div class="quality-layout"><div id="quality-list" class="quality-list" aria-label="候选维度"></div><section id="quality-detail" class="quality-detail" aria-label="维度证据"></section></div>
<p class="muted">“待 drop”勾选只保存在当前浏览器；可导出 CSV 后确定最终名单。<a href="candidates.csv">下载全部候选与统计</a> · <a href="candidates.json">方法及来源 JSON</a></p>'''
    document = render_page('维度质量审阅', 'quality', content, snapshot,
                           '<script src="quality.js" defer></script>')
    document = document.replace('</head>', '<link rel="stylesheet" href="quality.css"></head>')
    (target / 'index.html').write_text(document)
    for name in ['quality.css', 'quality.js']:
        shutil.copyfile(TEMPLATES / name, target / name)
    return counts
