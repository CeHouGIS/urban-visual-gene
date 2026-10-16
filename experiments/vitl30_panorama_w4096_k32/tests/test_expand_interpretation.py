import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from sae_experiments import panorama_vitl30_expand_interpretation as expansion


def test_sparse_peaks_skip_existing_dimensions_and_keep_first_tied_patch():
    # Two panoramas, three patch rows each. Repeated values must keep the
    # earliest patch; unrequested dimensions must never enter the ranking.
    values = np.array([4, 9, 4, 2, 6, 1], np.float32)
    dims = np.array([0, 1, 0, 2, 0, 2], np.uint16)
    ptr = np.array([0, 2, 3, 4, 5, 5, 6], np.uint32)
    peaks, patches = expansion.panorama_peaks(values, dims, ptr, 2,
        np.array([True, False, True]), width=3, tokens=3)
    np.testing.assert_array_equal(peaks, [[4, 0, 2], [6, 0, 1]])
    np.testing.assert_array_equal(patches, [[0, 0, 2], [0, 0, 2]])


def test_chunk_merging_matches_global_ranking_and_distinct_panorama_ties():
    peaks = np.array([[2, 0], [5, 1], [5, 0], [3, 2], [5, 1]], np.float32)
    positions = np.arange(10, dtype=np.int16).reshape(5, 2)
    state = (np.zeros((3, 2), np.float32), np.full((3, 2), -1, np.int64),
             np.zeros((3, 2), np.int16))
    for offset, stop in [(0, 2), (2, 5)]:
        state = expansion.merge_top(*state, peaks[offset:stop], positions[offset:stop], offset, 3)
    expected = np.argsort(-peaks, axis=0, kind='stable')[:3]
    np.testing.assert_array_equal(state[1], expected)
    np.testing.assert_array_equal(state[0], np.take_along_axis(peaks, expected, axis=0))
    np.testing.assert_array_equal(state[2], np.take_along_axis(positions, expected, axis=0))


def test_crop_reconstructs_winning_heading_without_losing_sparse_patch_positions(tmp_path):
    folder = tmp_path / 'activations' / 'Test__City' / '000000'
    folder.mkdir(parents=True)
    # One positive in heading 0, two in heading 90. The row pointers contain
    # many repeated entries from empty patch rows.
    patch_rows = [2, 784 + 3, 784 + 50]
    ptr = np.zeros(3137, np.uint32)
    for row in patch_rows:
        ptr[row + 1:] += 1
    for key, array in [('values', np.array([1, 4, 2], np.float16)),
                       ('indices', np.array([7, 7, 7], np.uint16)), ('indptr', ptr)]:
        np.save(folder / f'{key}.npy', array)
    image = tmp_path / 'direction90.png'
    Image.new('RGB', (84, 56)).save(image)
    frame = pd.DataFrame([dict(city='Test/City', cache_pano_index=0, panoid='pano',
                              image_90=str(image), sae_split='train', prototype_split='train')])
    result = expansion.crop_record(SimpleNamespace(cache=tmp_path), {'chunk_panos': 64},
        frame, feature=7, rank=1, pid=0, position=784 + 3, activation=4)
    heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(result['heatmap_png'])))).ravel()
    assert result['heading'] == 90 and result['native_view_size'] == [84, 56]
    assert heat[3] == 255 and heat[50] == 128 and np.count_nonzero(heat) == 2
    assert result['sae_split'] == 'train'


def test_selection_excludes_previously_interpreted_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion, 'WIDTH', 3)
    (tmp_path / 'identity.json').write_text(json.dumps(
        {'identity': 'baseline', 'checkpoint_sha256': expansion.CHECKPOINT}))
    records = {'identity': 'baseline', 'dimensions': [
        {'feature': 0, 'status': 'interpreted'}, {'feature': 1, 'status': 'unobserved_in_sample'},
        {'feature': 2, 'status': 'response_invalid'}]}
    path = tmp_path / 'dimension_interpretations.json'
    path.write_text(json.dumps(records))
    assert expansion.missing_features(tmp_path) == ([1, 2], 'baseline')
    records['identity'] = 'different'
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='identity differs'):
        expansion.missing_features(tmp_path)


def test_supplement_reports_only_requested_dimensions_and_rejects_cpu(tmp_path, monkeypatch):
    from sae_experiments import panorama_vitl30_dimension_vlm as vlm
    monkeypatch.setattr(vlm, 'WIDTH', 3)
    source, report, model = [tmp_path / name for name in ('source', 'report', 'model')]
    source.mkdir(); model.mkdir()
    (model / 'model.safetensors').write_bytes(b'fixture')
    (model / 'config.json').write_text('{}')
    run = dict(checkpoint_sha256='fixture', cities=['Test/City'], panoramas=50, images=200,
               scope='Expanded post-hoc corpus including training images', ranking='distinct panoramas',
               sample_kind='expanded_posthoc', target_features=[1, 2], target_dimensions=2)
    (source / 'run.json').write_text(json.dumps(run))
    row = dict(feature=1, rank=1, activation=2, pano_index=8, city='Test/City')
    (source / 'crops.jsonl').write_text(json.dumps(row) + '\n')
    args = SimpleNamespace(source=source, report=report, model=model, max_new_tokens=512,
                           input_mode='masked-top10', mask_fraction=.5)
    vlm.prepare(args)
    vlm.report(args)
    saved = lambda name: json.loads((report / name).read_text())
    assert saved('records/f0000.json')['status'] == 'not_requested'
    assert saved('records/f0002.json')['status'] == 'unobserved_in_sample'
    assert not (report / 'records/f0001.json').exists()
    summary = saved('summary.json')
    assert summary['requested_dimensions'] == 2
    assert summary['completed_or_explicitly_unobserved'] == 1
    assert summary['status_counts'] == {'not_requested': 1, 'pending': 1, 'unobserved_in_sample': 1}
    assert '不是独立验证集' in (report / 'REPORT.md').read_text()
    assert summary['source_scope']['target_features'] == [1, 2]
    with pytest.raises(ValueError, match='CPU inference/offloading is disabled'):
        vlm.worker(SimpleNamespace(device='cpu'))

    row['feature'] = 0
    (source / 'crops.jsonl').write_text(json.dumps(row) + '\n')
    args.report = tmp_path / 'another-report'
    with pytest.raises(ValueError, match='excluded dimension'):
        vlm.prepare(args)
