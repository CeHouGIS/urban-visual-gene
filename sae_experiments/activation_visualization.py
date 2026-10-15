"""Render relative SAE responses visibly without changing the underlying activations."""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


def activation_overlay(image, activation, scale):
    values = np.clip(np.asarray(activation, dtype=np.float32) / max(float(scale), 1e-12), 0, 1)
    field = np.asarray(Image.fromarray(values).resize(image.size, Image.Resampling.BILINEAR))
    color = colormaps['inferno'](field)[..., :3] * 255
    # Original images are always saved separately. A subdued grey background
    # makes the activated regions distinguishable from red cars or green trees.
    grey = np.asarray(image.convert('L'), dtype=np.float32)[..., None]
    base = np.repeat(grey * .53, 3, axis=2)
    alpha = (np.power(field, .4) * .96)[..., None]
    rendered = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(rendered)


def mark_peak(image, box):
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=(0, 0, 0), width=3)
    draw.rectangle(box, outline=(90, 245, 255), width=2)


def add_legend(panorama, raw_max):
    canvas = Image.new('RGB', (panorama.width, panorama.height + 52), (30, 30, 34))
    canvas.paste(panorama, (0, 0))
    gradient = np.tile(np.linspace(0, 1, 300, dtype=np.float32), (16, 1))
    bar = Image.fromarray((colormaps['inferno'](gradient)[..., :3] * 255).astype(np.uint8))
    canvas.paste(bar, (290, panorama.height + 11))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((12, panorama.height + 9), 'Relative group activation', fill='white', font=font)
    draw.text((273, panorama.height + 8), '0', fill='white', font=font)
    draw.text((596, panorama.height + 8), '1', fill='white', font=font)
    draw.text((645, panorama.height + 9), f'Peak = {raw_max:.4f} | Scale within panorama | Cyan box: retrieved patch', fill='white', font=font)
    return canvas
