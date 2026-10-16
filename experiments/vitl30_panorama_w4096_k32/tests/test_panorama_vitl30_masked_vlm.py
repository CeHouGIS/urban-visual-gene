import base64
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from sae_experiments.panorama_vitl30_dimension_vlm import (
    contact_sheet, mask_native, materialize_masked, prepare,
)


def test_native_mask_retains_separate_regions_and_holes_without_expansion():
    rgb = np.arange(56 * 84 * 3, dtype=np.uint8).reshape(56, 84, 3)
    heat = np.zeros((28, 28), np.uint8)
    heat[2:5, 3:6] = 255
    heat[3, 4] = 0  # enclosed hole must remain masked
    heat[10, 12] = 128  # separate high region must survive
    heat[8, 8] = 127  # just below the explicit quantized threshold
    clean, masked, mask, info = mask_native(Image.fromarray(rgb), heat)
    expected = np.repeat(np.repeat(heat >= 128, 2, axis=0), 3, axis=1)
    x0, y0, x1, y1 = info['bbox_native_view']
    expected = expected[y0:y1, x0:x1]
    np.testing.assert_array_equal(np.asarray(mask) > 0, expected)
    np.testing.assert_array_equal(np.asarray(masked)[expected], np.asarray(clean)[expected])
    assert np.all(np.asarray(masked)[~expected] == 127)
    assert info['kept_patches'] == 9


def test_invalid_maps_are_rejected_instead_of_fabricating_regions():
    with pytest.raises(ValueError, match='peak-normalized'):
        mask_native(Image.new('RGB', (50, 50)), np.zeros((28, 28), np.uint8))


def test_masked_inputs_preserve_original_geometry_and_auditable_native_pixels(tmp_path):
    source = tmp_path / 'source'
    report = tmp_path / 'report'
    source.mkdir()
    image = source / 'original.png'
    Image.new('RGB', (84, 56), (10, 200, 40)).save(image)
    heat = np.zeros((28, 28), np.uint8)
    heat[3, 3] = 255
    heat[6, 6] = 255
    buffer = io.BytesIO()
    Image.fromarray(heat).save(buffer, format='PNG')
    row = {'feature': 12, 'rank': 1, 'activation': .8, 'image': str(image),
           'native_view_size': [84, 56], 'heatmap_png': base64.b64encode(buffer.getvalue()).decode()}
    [selected] = materialize_masked(source, [row], report, .5)
    with Image.open(source / selected['crop']) as masked, Image.open(source / selected['mask']) as mask:
        assert np.all(np.asarray(masked)[np.asarray(mask) == 0] == 127)
    target = report / 'sheet.png'
    contact_sheet(source, [selected], target, tile=224, columns=5)
    assert target.exists()
    with Image.open(target) as sheet:
        pixels = np.asarray(sheet)
        assert np.any(np.all(pixels == (10, 200, 40), axis=-1))
        assert np.any(np.all(pixels == 127, axis=-1))
    audit = json.loads((report / 'masked_crops/f0012/r01.json').read_text())
    assert audit['mask_fraction'] == .5
    assert audit['image'] == str(image)
    assert len(audit['pixel_mask_sha256']) == 64


def test_prepare_selects_only_first_ten_and_records_identity(tmp_path):
    source, report, model = [tmp_path / x for x in ('source', 'report', 'model')]
    source.mkdir(); model.mkdir()
    (model / 'model.safetensors').write_bytes(b'test')
    (model / 'config.json').write_text('{}')
    (source / 'run.json').write_text(json.dumps({'checkpoint_sha256': 'abc',
        'cities': ['test'], 'panoramas': 11, 'images': 44,
        'scope': 'fixture', 'ranking': 'distinct panoramas'}))
    rows = [dict(feature=0, rank=i + 1, activation=11 - i, pano_index=i,
                 crop='unused.png', city='test') for i in range(11)]
    (source / 'crops.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    args = SimpleNamespace(source=source, report=report, model=model, max_new_tokens=512,
                           input_mode='masked-top10', mask_fraction=.5)
    prepare(args)
    evidence = json.loads((report / 'evidence.json').read_text())['dimensions']
    assert len(evidence) == 4096
    assert [x['rank'] for x in evidence[0]['crops']] == list(range(1, 11))
    assert evidence[0]['sheets'] == ['inputs/f0000_0.png']
    assert evidence[1]['count'] == 0
    identity = json.loads((report / 'identity.json').read_text())
    assert identity['evidence']['mode'] == 'masked-top10'
    assert identity['evidence']['minimum_evidence'] == 5
    args.mask_fraction = .7
    with pytest.raises(ValueError, match='identity changed'):
        prepare(args)
