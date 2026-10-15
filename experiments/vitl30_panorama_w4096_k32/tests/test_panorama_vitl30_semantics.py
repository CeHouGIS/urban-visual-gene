import hashlib
import json

import numpy as np
import pytest
import torch

from sae_experiments.panorama_vitl30_semantics import (
    checked_visual_review, complete_response, diverse_examples, parse_response, sparse_group_peaks,
)


def test_retrieval_keeps_non_dominant_sparse_responses():
    # Group 1 never wins a patch, but it has a positive sparse response.
    acts = torch.tensor([[.9, .0, .3], [.8, .2, .1]])
    labels = np.array([0, 0, 1])
    rows = sparse_group_peaks(acts, labels, 'X/Y', 7, 'pano')
    assert len(rows) == 2
    assert rows[1]['group'] == 1
    assert rows[1]['latent'] == 2
    assert rows[1]['patch'] == 0
    assert rows[1]['score'] == pytest.approx(.3)
    zeros = sparse_group_peaks(torch.zeros_like(acts), labels, 'X/Y', 7, 'pano')
    assert zeros == []


def test_retry_rejects_truncated_and_wrong_schema_responses():
    raw = '{"label":"tree","semantic_type":"vegetation","recurring_cues":["leaves"],"ambiguities":[],"confidence":"medium"}'
    parsed = parse_response(raw)
    record = {'identity': 'abc', 'status': 'interpreted', 'raw_response': raw,
              'interpretation': parsed, 'truncated': False}
    assert complete_response(record, 'abc')
    assert not complete_response(record, 'other')
    assert not complete_response({**record, 'truncated': True}, 'abc')
    assert not complete_response({**record, 'raw_response': raw[:-1]}, 'abc')
    with pytest.raises(ValueError):
        parse_response(raw.replace('vegetation', 'artifact'))


def test_evidence_diversity_keeps_distinct_panoramas_and_cities():
    rows = [{'city': city, 'pano_index': pid, 'latent': latent, 'score': score}
            for city, pid, latent, score in [('A/B', 1, 4, .9), ('A/B', 1, 5, .8),
                                           ('A/B', 2, 6, .7), ('C/D', 3, 4, .6),
                                           ('E/F', 4, 7, .5)]]
    chosen = diverse_examples(rows, 3)
    assert len({(x['city'], x['pano_index']) for x in chosen}) == 3
    assert len({x['city'] for x in chosen}) == 3


def test_visual_review_rejects_changed_crop_pixels(tmp_path, monkeypatch):
    import sae_experiments.panorama_vitl30_semantics as semantics

    monkeypatch.setattr(semantics, 'GROUPS', 1)
    crop = tmp_path / 'crop.jpg'
    crop.write_bytes(b'original reviewed pixels')
    source = {'city': 'A/B', 'pano_index': 7, 'latent': 12, 'focused_crop': str(crop)}
    entry = {'group': 0, 'evidence': [{**source, 'sha256': hashlib.sha256(crop.read_bytes()).hexdigest()}]}
    review = {'identity': 'abc', 'independent_human_validation': False, 'groups': [entry]}
    (tmp_path / 'visual_review.json').write_text(json.dumps(review))
    manifest = {'groups': {'0': [source]}}
    entries, provenance = checked_visual_review(tmp_path, 'abc', manifest)
    assert entries[0] == entry
    assert provenance['independent_human_validation'] is False
    crop.write_bytes(b'different pixels')
    with pytest.raises(ValueError, match='Changed reviewed crop pixels'):
        checked_visual_review(tmp_path, 'abc', manifest)
