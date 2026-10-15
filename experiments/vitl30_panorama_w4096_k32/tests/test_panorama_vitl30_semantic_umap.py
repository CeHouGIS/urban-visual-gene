"""Check held-out provenance and clean, deduplicated semantic crops."""
import base64
import io

import numpy as np
from PIL import Image

from sae_experiments.panorama_vitl30_semantic_umap import heldout_ids, save_crops


def test_sample_is_exact_training_holdout():
    import hashlib
    city = 'US/NewYork'
    seed = int.from_bytes(hashlib.sha256(f'42/{city}'.encode()).digest()[:8], 'little')
    training_split = np.random.default_rng(seed).choice(np.arange(3000), 128, replace=False)
    np.testing.assert_array_equal(heldout_ids(city, 3000), np.sort(training_split))
    assert len(set(heldout_ids(city, 3000))) == 128


def test_crops_share_identical_pixels_but_keep_dimension_heatmaps(tmp_path):
    source = tmp_path / 'original.png'
    Image.new('RGB', (448, 448), (13, 101, 207)).save(source)
    row = {'image': str(source), 'city': 'Test', 'pano': 'P', 'heading': 90}
    heat = np.zeros((28, 28), np.float32)
    heat[10:12, 10:12] = 2
    heat[10, 10] = 4
    second = heat.copy()
    second[11, 11] = 3
    selections = [{'feature': j, 'rank': 1, 'image_id': 1, 'pano_index': 0,
                   'activation': 4.0} for j in [2, 17]]
    crops = save_crops((1, row, selections, [heat, second], str(tmp_path)))
    assert crops[0]['crop'] == crops[1]['crop']
    assert crops[0]['heatmap_png'] != crops[1]['heatmap_png']
    with Image.open(tmp_path / crops[0]['crop']) as clean:
        np.testing.assert_allclose(np.array(clean).mean(axis=(0, 1)), [13, 101, 207], atol=3)
    with Image.open(io.BytesIO(base64.b64decode(crops[0]['heatmap_png']))) as encoded:
        assert encoded.size == (28, 28)
        assert np.array(encoded).max() == 255
