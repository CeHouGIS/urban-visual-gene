"""Render the two-level Qwen taxonomy as a reviewable static figure."""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = (
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
)


def font(size):
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError('A CJK font is needed to export the semantic map PNG')


def activation_thumbnail(example, site, size=132):
    source = site / 'semantic-map' / example['image']
    with Image.open(source) as original:
        base = original.convert('RGBA').resize((size, size), Image.Resampling.LANCZOS)
    heat = Image.open(BytesIO(base64.b64decode(example['heat']))).convert('L')
    if heat.size != (28, 28):
        raise ValueError(f'Unexpected heatmap grid for f{example["feature"]}: {heat.size}')
    rgba = Image.new('RGBA', heat.size)
    rgba.putdata([(255, round(210 * min(1, value / 255 * 1.5)), 30,
                   round(225 * max(0, (value / 255 - .08) / .92)))
                  for value in heat.tobytes()])
    return Image.alpha_composite(base, rgba.resize((size, size), Image.Resampling.NEAREST)).convert('RGB')


def render(site, output):
    data = json.loads((site / 'semantic-map/map.json').read_text())
    leaves = [feature for family in data['families'] for child in family['children']
              for feature in child['features']]
    if (data['family_count'] != len(data['families']) or data['leaf_count'] != 4096
            or sorted(leaves) != list(range(4096))):
        raise ValueError('Hierarchy counts do not cover the main SAE')
    child_heights = [[max(28, 2 * len(child['features'])) for child in family['children']]
                     for family in data['families']]
    family_heights = [max(178, sum(parts) + 20) for parts in child_heights]
    width, header, footer = 1740, 175, 95
    height = header + sum(family_heights) + footer
    image = Image.new('RGB', (width, height), '#f7f9f6')
    draw = ImageDraw.Draw(image)
    heading, parent_font, small, number = font(34), font(19), font(14), font(12)
    draw.text((40, 24), 'SAE feature hierarchy', font=heading, fill='#17342e')
    draw.text((40, 76), f'{data["family_count"]} parent classes  /  '
              f'{sum(len(f["children"]) for f in data["families"])} subclasses  /  '
              f'4,096 individual feature leaves', font=small, fill='#526d63')
    for x, title in ((75, 'Parent class'), (690, 'Subclass / dimensions'),
                     (948, 'Feature leaves'), (1110, 'SAE activation evidence (4 each)')):
        draw.text((x, 137), title, font=small, fill='#526d63')
    root_x, parent_x, child_x, leaf_x, leaf_end = 52, 340, 682, 954, 1018
    parent_centers = []
    y = header
    for family, block_height, parts in zip(data['families'], family_heights, child_heights):
        center = y + block_height / 2
        parent_centers.append(center)
        draw.line((66, y, width - 40, y), fill='#dbe6df', width=2)
        cursor = y + (block_height - sum(parts)) / 2
        child_centers = []
        for child_index, (child, part_height) in enumerate(zip(family['children'], parts)):
            features = child['features']
            child_center = cursor + part_height / 2
            child_centers.append(child_center)
            leaf_ys = [cursor + (index + .5) * part_height / len(features)
                       for index in range(len(features))] if features else []
            if leaf_ys:
                draw.line((child_x, child_center, leaf_x, child_center), fill='#a9beb0', width=1)
                draw.line((leaf_x, leaf_ys[0], leaf_x, leaf_ys[-1]), fill='#668877', width=1)
                for leaf_y in leaf_ys:
                    draw.line((leaf_x, leaf_y, leaf_end, leaf_y), fill='#526e5e', width=1)
            draw.rectangle((child_x + 10, child_center - 10, child_x + 105, child_center + 10),
                           fill='#f7f9f6')
            draw.text((child_x + 15, child_center - 9),
                      f'{child_index + 1:02d}  n={child["count"]}', font=number, fill='#315a46')
            cursor += part_height
        if child_centers:
            draw.line((parent_x, child_centers[0], parent_x, child_centers[-1]),
                      fill='#779a83', width=2)
            for child_center in child_centers:
                draw.line((parent_x, child_center, child_x, child_center),
                          fill='#779a83', width=1)
        draw.line((root_x, center, parent_x, center), fill='#386b50', width=2)
        words = family['label_en'].split()
        lines, line = [], ''
        for word in words:
            candidate = f'{line} {word}'.strip()
            if line and draw.textlength(candidate, font=parent_font) > 248:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        for line_index, text in enumerate(lines):
            draw.text((75, center - 76 + line_index * 23), text,
                      font=parent_font, fill='#213b30')
        draw.text((75, center - 20),
                  f'{family["count"]:,} features  /  {len(family["children"])} subclasses',
                  font=small, fill='#526d63')
        if family['id'] == 'unresolved':
            draw.text((75, center + 7),
                      f'Weak {data["weak_count"]} + no activation {data["zero_count"]}',
                      font=number, fill='#876330')
        for example_index, example in enumerate(family['examples']):
            left = 1110 + example_index * 145
            image.paste(activation_thumbnail(example, site), (left, round(center - 68)))
            draw.text((left + 2, center + 66), f'f{example["feature"]}',
                      font=number, fill='#344e43')
        y += block_height
    draw.line((root_x, parent_centers[0], root_x, parent_centers[-1]),
              fill='#386b50', width=3)
    draw.line((66, height - footer, width - 40, height - footer), fill='#dbe6df', width=2)
    draw.text((40, height - 71),
              'One terminal tick per SAE dimension (f0-f4095). Branch length is schematic, not a semantic distance.',
              font=small, fill='#526d63')
    draw.text((40, height - 45),
              'Qwen labels and spatial-encoding assignments are provisional; no-positive-activation features have no image evidence.',
              font=small, fill='#526d63')
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    site = args.site_dir.resolve()
    print(render(site, args.output or site / 'semantic-map/semantic-hierarchy.png'))
