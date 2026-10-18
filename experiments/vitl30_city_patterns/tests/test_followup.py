import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.vitl30_city_patterns.code import followup
from experiments.vitl30_city_patterns.code.extract import save_chunk


def fixture(tmp_path):
    selection = tmp_path / 'selection'
    extraction = tmp_path / 'extraction'
    records = []
    cities = []
    for city, panoids in [('Country/A', ['a0', 'a1']), ('Country/B', ['b0'])]:
        path = selection / city / 'selected.parquet'
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table({'panoid': panoids, 'lat': [1.0] * len(panoids),
                                 'lon': [2.0] * len(panoids),
                                 'grid_x': [12] * len(panoids),
                                 'grid_y': [34] * len(panoids)}), path)
        digest = followup.sha256(path)
        records.append({'city': city, 'selected_panoramas': len(panoids),
                        'selected_sha256': digest, 'epsg': 32631})
        cities.append({'city': city, 'selection_rows': len(panoids),
                       'selection_sha256': digest})
    (selection / 'manifest.json').write_text(json.dumps({'cities': records}))
    extraction.mkdir()
    (extraction / 'run.json').write_text(json.dumps({
        'version': 3, 'identity': 'test-run', 'cell_meters': 1000,
        'max_panoramas_per_cell': 10, 'rows_per_chunk': 2, 'cities': cities}))
    for shard, n in enumerate([2, 1, 0, 0]):
        (extraction / f'progress_shard{shard}.json').write_text(json.dumps({
            'shard': shard, 'shards': 4, 'status': 'complete', 'candidates': n}))
    composition = np.ones(4096, dtype=np.float16)
    first = np.zeros((4, 64), dtype=np.float16)
    second = first.copy()
    first[:, 0] = 2
    second[:, 1] = 2
    save_chunk(extraction / 'Country__A/000000', 'test-run', 0, 2, 2,
               [('a0', 1., 2., 12, 34, composition, first),
                ('a1', 1., 2., 12, 34, composition, second)],
               {'missing': 0, 'black': 0, 'tunnel': 0, 'unreadable': 0})
    save_chunk(extraction / 'Country__B/000000', 'test-run', 0, 1, 1,
               [('b0', 1., 2., 12, 34, composition, second)],
               {'missing': 0, 'black': 0, 'tunnel': 0, 'unreadable': 0})
    return selection, extraction


def test_followup_gates_complete_shards_and_chunks(tmp_path):
    selection, extraction = fixture(tmp_path)
    assert followup.ready(extraction, selection, 2) == (True, 'four shards complete')
    marker = extraction / 'Country__B/000000/complete.json'
    marker.unlink()
    with pytest.raises(ValueError, match='Missing completed chunk'):
        followup.analyze(extraction, selection, tmp_path / 'out', 2)


def test_queue_tracks_current_and_dependent_work(tmp_path):
    selection, extraction = fixture(tmp_path)
    waiting = followup.queue_snapshot(extraction, selection, 'waiting', 'extracting')
    tasks = {item['id']: item for item in waiting['tasks']}
    assert waiting['selected_panoramas'] == waiting['extraction_candidates_done'] == 3
    assert tasks['extract']['status'] == 'running'
    assert tasks['city_cell_screening']['status'] == 'waiting'
    assert tasks['road_sides']['status'] == 'blocked'
    done = followup.queue_snapshot(extraction, selection, 'complete', 'done')
    tasks = {item['id']: item for item in done['tasks']}
    assert tasks['extract']['status'] == tasks['city_cell_screening']['status'] == 'complete'
    assert tasks['chapter5_figures']['status'] == 'pending'
    assert tasks['chapter5_figures']['depends_on'] == ['prototype_city_distribution']


def test_followup_emits_cell_weighted_summary_and_pending_tasks(tmp_path):
    selection, extraction = fixture(tmp_path)
    out = tmp_path / 'out'
    result = followup.analyze(extraction, selection, out, 2)
    assert result['processed'] == 3 and result['occupied_cells'] == 2
    assert result['valid_cells'] == 2 and result['zero_mass_panoramas'] == 0
    cells = pq.read_table(out / 'cells.parquet').to_pylist()
    assert len(cells) == 2
    assert cells[0]['group_00'] == .5 and cells[0]['group_01'] == .5
    assert cells[1]['group_00'] == 0 and cells[1]['group_01'] == 1
    tasks = json.loads((out / 'next_tasks.json').read_text())['tasks']
    assert {task['id'] for task in tasks} == {
        'candidate_combinations', 'road_sides', 'prototype_city_distribution', 'chapter5_figures'}
    assert all(task['status'] == 'pending' for task in tasks)
    assert (out / 'city_group_mass_preview.png').stat().st_size > 10000
    assert followup.analyze(extraction, selection, out, 2) == result
