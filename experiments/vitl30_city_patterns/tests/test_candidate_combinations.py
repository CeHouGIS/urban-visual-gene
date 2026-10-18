import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from experiments.vitl30_city_patterns.code import candidate_combinations as cc


def hierarchy():
    return {'families': [{'id': 'building', 'counts_as_element': True},
                         {'id': 'tree', 'counts_as_element': True}],
            'dimensions': [dict(feature=f, family=family, counts_as_element=True,
                                provisional_graph_candidate=eligible)
                           for f, family, eligible in [(0, 'building', True),
                                                       (1, 'tree', True),
                                                       (2, 'building', True),
                                                       (3, 'tree', False)]]}


def test_parts_merge_and_unknown_mass_remains_in_denominator():
    dimensions = cc.family_dimensions(hierarchy())
    assert dimensions['building'].tolist() == [0, 2]
    values = np.zeros((2, 4096), dtype=np.float32)
    values[0, [0, 1, 2, 4095]] = [1, 2, 3, 14]
    scores = cc.family_shares(values, dimensions)
    assert scores[0, 0] == pytest.approx(.2)
    assert scores[0, 1] == pytest.approx(.1)
    assert not scores[1].any()


def test_hypergeometric_test_matches_exact_binary_shuffle_distribution():
    p, expected = cc.permutation_test(np.array([4]), np.array([2]), np.array([2]),
                                      2, 20000, np.random.default_rng(8))
    assert .15 < p < .18  # P(overlap=2) = 1 / 6.
    assert expected == 1
    p, expected = cc.permutation_test(np.array([4]), np.array([0]), np.array([2]),
                                      0, 99, np.random.default_rng(8))
    assert p == 1 and expected == 0


def make_events(concentrated):
    names = [f'city_{i}' for i in range(30)]
    selected = ([c for c in names if cc.stable_city(c) % 3 != 0][:3] +
                [c for c in names if cc.stable_city(c) % 3 == 0][:3])
    events = {}
    for city in selected:
        n = 200
        present = np.zeros((n, len(cc.ELEMENTS)), dtype=bool)
        response = np.arange(n) < 100 if concentrated else np.arange(n) % 2 == 0
        present[:, 0] = response
        present[:, 1] = response
        events[city] = {'panoid': np.arange(n), 'present': present,
                        'cell_index': np.arange(n) // 10,
                        'block_index': np.arange(n) // 50}
    return events


def test_spatial_null_rejects_spurious_city_level_association():
    events = make_events(concentrated=True)
    discovery, confirmation, results = cc.screen(events, permutations=999)
    row = results[0]
    assert row['p_city'] < .01
    assert row['p_spatial'] == 1
    assert row['status'] == 'not_confirmed'
    assert row['discovery_cells'] == 30  # 1-km cells, not six 5-km blocks.
    n, _, _ = cc.strata(events, confirmation, (0, 1), False)
    assert n.sum() == 600  # Includes panoramas with no eligible elements.


def test_recurring_association_confirms_in_held_out_cities():
    _, _, results = cc.screen(make_events(concentrated=False), permutations=999)
    assert len(results) == 1
    assert results[0]['status'] == 'co_presence_candidate'
    assert results[0]['q_both'] < .05
    assert results[0]['layout_evidence'] == 'unavailable_from_pooled_vectors'
    _, _, too_few_cells = cc.screen(make_events(False), permutations=99, min_cells=100)
    assert too_few_cells == []


def test_local_stage_writes_verifiable_outputs_and_resumes_caches(tmp_path, monkeypatch):
    extraction = tmp_path / 'extraction'
    followup = tmp_path / 'followup'
    output = followup / 'stages/candidate_combinations'
    extraction.mkdir()
    followup.mkdir()
    cities = [f'country/city_{i}' for i in range(30)]
    (extraction / 'run.json').write_text(json.dumps({
        'identity': 'test-run', 'cities': [{'city': c} for c in cities]}))
    (followup / 'complete.json').write_text(json.dumps({
        'identity': {'extract_run_identity': 'test-run'}, 'cities': 30, 'processed': 240}))
    labels = tmp_path / 'hierarchy.json'
    labels.write_text(json.dumps(hierarchy()))
    for city in cities:
        folder = extraction / city.replace('/', '__') / '000000'
        folder.mkdir(parents=True)
        comp = np.zeros((8, 4096), dtype=np.float16)
        comp[:, 0] = 1
        archive = folder / 'vectors.npz'
        np.savez_compressed(archive, composition=comp, panoid=np.array([f'p{i}' for i in range(8)]),
                            grid_x=np.arange(8), grid_y=np.zeros(8, dtype=np.int32))
        (folder / 'complete.json').write_text(json.dumps({
            'identity': 'test-run', 'archive_bytes': archive.stat().st_size,
            'processed_panoramas': 8}))
    args = SimpleNamespace(extraction=extraction, followup=followup, output=output,
                           hierarchy=labels, run_identity='test-run', threshold=.02, permutations=99)
    cc.run(args)
    assert pq.read_table(output / 'panorama_elements.parquet').num_rows == 240
    assert json.loads((output / 'summary.json').read_text())['candidate_count'] == 0
    assert json.loads((output / 'complete.json').read_text())['run_identity'] == 'test-run'
    monkeypatch.setattr(cc, 'family_shares', lambda *a: pytest.fail('must reuse local city cache'))
    cc.run(args)
