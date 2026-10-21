"""Reject model mixups and misleading real-image evidence before publication."""
import copy
import json

import pytest

from docs.site.prototype_gallery import load_gallery


def fixture_gallery(tmp_path):
    protocol = {'identity': 'current-experiment', 'checkpoint_sha256': 'main-sae',
                'model': {'scales': [3], 'per_scale': 2}}
    identity = {'experiment_identity': protocol['identity'], 'model_sha256': 'current-model',
                'sae_sha256': 'main-sae', 'split': 'dev', 'test_used': False}
    records = [{'prototype': p, 'scale': 3, 'identity': identity, 'model_sha256': 'current-model',
                'instances': []} for p in range(2)]
    records[0]['instances'] = [{'pano_row': 10, 'coefficient_fp32': 2., 'removing_response_delta_sse': 1.}]
    bundle = {'identity': identity, 'prototypes': records}
    folder = tmp_path / 'outputs/gallery'
    folder.mkdir(parents=True)
    (folder / 'complete.json').write_text(json.dumps({'identity': identity, 'examples': 1, 'templates_with_examples': 1}))
    return protocol, bundle, folder


def write_bundle(folder, bundle):
    (folder / 'prototypes.json').write_text(json.dumps(bundle))


def test_retains_inactive_slots(tmp_path):
    protocol, bundle, folder = fixture_gallery(tmp_path)
    write_bundle(folder, bundle)
    result = load_gallery(tmp_path, protocol, 'current-model')
    assert len(result[2]['prototypes']) == 2
    assert result[2]['prototypes'][1]['instances'] == []


def test_older_model_cannot_be_used_under_current_ids(tmp_path):
    protocol, bundle, folder = fixture_gallery(tmp_path)
    write_bundle(folder, bundle)
    with pytest.raises(ValueError, match='another model'):
        load_gallery(tmp_path, protocol, 'older-model')


@pytest.mark.parametrize('change', ['harmful', 'duplicate', 'missing_slot', 'mixed_record'])
def test_rejects_invalid_exemplars_and_mixed_models(tmp_path, change):
    protocol, bundle, folder = fixture_gallery(tmp_path)
    if change == 'harmful':
        bundle['prototypes'][0]['instances'][0]['removing_response_delta_sse'] = -1
    elif change == 'duplicate':
        bundle['prototypes'][0]['instances'] *= 2
    elif change == 'missing_slot':
        bundle['prototypes'].pop()
    else:
        bundle['prototypes'][0]['identity'] = copy.deepcopy(bundle['identity'])
        bundle['prototypes'][0]['identity']['model_sha256'] = 'older-model'
    write_bundle(folder, bundle)
    with pytest.raises(ValueError):
        load_gallery(tmp_path, protocol, 'current-model')
