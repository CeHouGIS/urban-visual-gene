from pathlib import Path

import numpy as np
import torch
from PIL import Image

from formal.analysis.build_expanded_sae_gallery import (
    render_panorama,
    save_sparse_map,
    scan_candidates,
    select_feature_ids,
)


def test_candidate_scan_keeps_distinct_panorama_maxima():
    class Model:
        latent_dim = 4

        def encode(self, x):
            return x

        def decode(self, x):
            return x

    features = np.zeros((256, 4), dtype=np.float16)
    features[7, 0] = 3
    features[128 + 9, 0] = 5
    features[128 + 11, 1] = 2
    support, values, rows, heldout_loss = scan_candidates(Model(), features, torch.device("cpu"), 128, 2)
    assert support.tolist() == [2, 1, 0, 0]
    assert sorted(rows[:, 0].tolist()) == [7, 137]
    assert sorted(values[:, 0].tolist()) == [3, 5]
    assert heldout_loss == 253 / 256


def test_four_view_image_and_sparse_map_are_aligned(tmp_path: Path):
    colors = ((250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0))
    paths = {}
    for heading, color in zip((0, 90, 180, 270), colors):
        path = tmp_path / f"{heading}.png"
        Image.new("RGB", (12, 12), color).save(path)
        paths[heading] = str(path)
    target = tmp_path / "panorama.jpg"
    render_panorama(paths, target)
    image = Image.open(target)
    assert image.size == (1792, 448)
    for position, color in enumerate(colors):
        actual = image.getpixel((position * 448 + 224, 224))
        assert max(abs(a - b) for a, b in zip(actual, color)) < 5

    grid = np.zeros((28, 112), dtype=np.float32)
    grid[3, 2] = 1
    grid[3, 30] = 2
    grid[3, 58] = 3
    grid[3, 86] = 4
    mask_path = tmp_path / "mask.png"
    save_sparse_map(grid, mask_path)
    alpha = np.asarray(Image.open(mask_path))[:, :, 3]
    assert alpha.shape == (28, 112)
    assert np.count_nonzero(alpha) == 4
    assert np.all(np.diff(alpha[3, [2, 30, 58, 86]]) > 0)


def test_feature_selection_is_reproducible_across_support_bands():
    support = np.arange(1, 101)
    selected = select_feature_ids(support, per_band=2, seed=12)
    assert selected == select_feature_ids(support, per_band=2, seed=12)
    assert len(selected) == 6 and len(set(selected)) == 6
    assert all(support[feature] >= 25 for feature in selected)
