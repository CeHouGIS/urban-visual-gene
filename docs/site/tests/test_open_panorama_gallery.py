"""Current model, full headings and development-only evidence gate publication."""
import hashlib
import json

import pytest

from docs.site import open_panorama_gallery as gallery


def evidence(tmp_path, monkeypatch):
    out = tmp_path / 'outputs'
    (out / 'gallery').mkdir(parents=True)
    (out / 'best.pt').write_bytes(b'current-prototype-checkpoint')
    (out / 'protocol.json').write_text(json.dumps({'identity': 'current'}))
    payload = {
        'identity': 'current', 'checkpoint_sha256': gallery.CHECKPOINT,
        'prototype_checkpoint_sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest(),
        'test_used': False, 'full_panorama': True, 'headings': [0,90,180,270],
        'native_grid': [28,112], 'prototypes': [{
            'id': 3, 'train_support': 20, 'train_cities': 3, 'dev_support': 4,
            'top_groups': [], 'examples': [{
                'image': 'p0003_example00.webp', 'partition': 'dev',
                'city': 'Country/City', 'panoid': 'abc', 'nmse': .4,
                'mask_fraction': 1., 'source_images': {'0': '/private/source.jpg'}}]}],
        'training_summary': {'events': [], 'dev': {
            'panoramas': 10, 'nmse_unit_normalized': .4,
            'group_pooled_patch_coverage_all_denominator': .85,
            'four_directions_each_at_least_80pct': .5}}}
    (out / 'gallery/p0003_example00.webp').write_bytes(b'image-asset')
    monkeypatch.setattr(gallery, 'OUT', out)
    return out, payload


def write(out, payload):
    (out / 'gallery/index.json').write_text(json.dumps(payload))


@pytest.mark.parametrize('invalid', ['checkpoint', 'test', 'headings', 'partition'])
def test_rejects_mixed_or_sealed_evidence(tmp_path, monkeypatch, invalid):
    out, payload = evidence(tmp_path, monkeypatch)
    if invalid == 'checkpoint':
        payload['prototype_checkpoint_sha256'] = 'old-model'
    elif invalid == 'test':
        payload['test_used'] = True
    elif invalid == 'headings':
        payload['headings'] = [0,180,90,270]
    else:
        payload['prototypes'][0]['examples'][0]['partition'] = 'test'
    write(out, payload)
    target = tmp_path / 'site'
    target.mkdir()
    with pytest.raises(ValueError):
        gallery.install(target, {'status': 'complete'})
    assert not list(target.rglob('*.webp'))


def test_removes_stale_images_and_private_paths(tmp_path, monkeypatch):
    out, payload = evidence(tmp_path, monkeypatch)
    write(out, payload)
    target = tmp_path / 'site'
    (target / 'open-panorama').mkdir(parents=True)
    (target / 'open-panorama/old.webp').write_bytes(b'stale')
    markup = gallery.install(target, {'status': 'complete'})
    public = json.loads((target / 'open-panorama/index.json').read_text())
    assert 'source_images' not in public['prototypes'][0]['examples'][0]
    assert not (target / 'open-panorama/old.webp').exists()
    assert '原4096维覆盖或语义解释覆盖' in markup
