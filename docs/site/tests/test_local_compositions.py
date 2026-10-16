"""The public gallery must not turn direction variants into new prototypes."""
import hashlib
import json

from docs.site import local_composition_gallery


def test_gallery_collapses_angular_variants_without_changing_tests(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    output = source / 'experiments/vitl30_local_compositions/outputs'
    output.mkdir(parents=True)
    (output / 'gallery').mkdir()
    hierarchy = source / 'experiments/vitl30_relational_prototypes/results/hierarchy.json'
    hierarchy.parent.mkdir(parents=True)
    hierarchy.write_text('{}')
    split = source / 'experiments/vitl30_prototype_coverage/outputs/protocol.json'
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({'identity': 'split-a'}))
    patterns = []
    for n, relation in enumerate(('cross_90_U', 'cross_270_U', 'same_UM')):
        filename = f'p{n}.webp'
        (output / 'gallery' / filename).write_bytes(b'example')
        patterns.append({'families': ['road', 'tree'], 'relation': relation,
                         'train_panoramas': 20, 'train_blocks': 20, 'train_cities': 4,
                         'dev_panoramas': 8, 'dev_blocks': 8, 'dev_cities': 4,
                         'null_means': {'cooccurrence': 2, 'layout': 3},
                         'q': {'cooccurrence': .02, 'layout': .03},
                         'examples': [{'city': 'X/Y', 'panoid': 'x', 'image': filename}]})
    data = {'status': 'complete', 'checkpoint_sha256': local_composition_gallery.CHECKPOINT,
            'split_identity': 'split-a', 'hierarchy_sha256': __import__('hashlib').sha256(b'{}').hexdigest(),
            'sampling': {'cities': 29, 'train_panoramas': 100, 'dev_panoramas': 40, 'sampled_images': 560},
            'scope': 'exploratory balanced city sample; sealed test not accessed',
            'findings': {'displayed': 3, 'all_candidates': 10, 'q_below_005_both_nulls': 2},
            'patterns': patterns}
    (output / 'results.json').write_text(json.dumps(data))
    monkeypatch.setattr(local_composition_gallery, 'ROOT', source)
    monkeypatch.setattr(local_composition_gallery, 'OUT', output)
    page = local_composition_gallery.install(tmp_path / 'site')
    public = json.loads((tmp_path / 'site/local-compositions/results.json').read_text())
    assert len(public['patterns']) == 2
    assert public['findings']['all_candidates'] == 10
    assert public['presentation']['significance_tests_recomputed'] is False
    assert '两个不同组合布局' not in page


def test_snapshot_builds_without_server_outputs(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    snapshot = source / 'experiments/vitl30_local_compositions/results'
    (snapshot / 'gallery').mkdir(parents=True)
    payload = {'status': 'complete', 'checkpoint_sha256': local_composition_gallery.CHECKPOINT,
               'split_identity': 'split-a', 'sampling': {'cities': 29, 'train_panoramas': 1,
               'dev_panoramas': 1, 'sampled_images': 8},
               'scope': 'exploratory balanced city sample; sealed test not accessed',
               'findings': {'displayed': 1, 'all_candidates': 1, 'q_below_005_both_nulls': 1},
               'patterns': [{'families': ['road', 'tree'], 'relation': 'cross_90_U',
               'train_panoramas': 1, 'train_blocks': 1, 'train_cities': 1,
               'dev_panoramas': 1, 'dev_blocks': 1, 'dev_cities': 1,
               'null_means': {'cooccurrence': 0, 'layout': 0},
               'q': {'cooccurrence': .01, 'layout': .01},
               'examples': [{'city': 'A/B', 'panoid': 'p', 'image': 'p.webp'}]}]}
    hierarchy = source / 'experiments/vitl30_relational_prototypes/results/hierarchy.json'
    hierarchy.parent.mkdir(parents=True)
    hierarchy.write_text('{}')
    payload['hierarchy_sha256'] = hashlib.sha256(hierarchy.read_bytes()).hexdigest()
    body = json.dumps(payload)
    (snapshot / 'results.json').write_text(body)
    (snapshot / 'gallery/p.webp').write_bytes(b'example')
    (snapshot / 'provenance.json').write_text(json.dumps({
        'results_sha256': hashlib.sha256(body.encode()).hexdigest(),
        'checkpoint_sha256': local_composition_gallery.CHECKPOINT,
        'split_identity': 'split-a', 'hierarchy_sha256': payload['hierarchy_sha256'],
        'gallery_sha256': {'p.webp': hashlib.sha256(b'example').hexdigest()}}))
    monkeypatch.setattr(local_composition_gallery, 'ROOT', source)
    monkeypatch.setattr(local_composition_gallery, 'OUT', source / 'experiments/vitl30_local_compositions/outputs')
    monkeypatch.setattr(local_composition_gallery, 'RESULT', snapshot)
    local_composition_gallery.install(tmp_path / 'site')
    assert (tmp_path / 'site/local-compositions/p.webp').read_bytes() == b'example'
    (snapshot / 'gallery/p.webp').write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError, match='image changed'):
        local_composition_gallery.install(tmp_path / 'site')
