import json

import numpy as np

from experiments.vitl30_city_patterns.code.extract import (
    eligible, image_paths, save_chunk)


def test_four_view_paths_and_metadata_gate():
    paths = image_paths('Germany/Hamburg', 'abcd')
    assert [path.name for path in paths] == [f'abcd_{h}.jpg' for h in (0, 90, 180, 270)]
    assert '/Germany/Hamburg/a/b/' in str(paths[0])
    assert eligible({'panoid': 'abcd', 'exists': True, '#images': 4}, 'exists', True)
    assert not eligible({'panoid': 'abcd', 'exists': True, '#images': 3}, 'exists', True)
    assert not eligible({'panoid': 'abcd', 'exists': False}, 'exists', False)


def test_chunk_archive_keeps_panoid_and_vector_alignment(tmp_path):
    first = np.ones(4096, np.float16)
    second = np.full(4096, 2, np.float16)
    head = np.ones((4, 64), np.float16)
    save_chunk(tmp_path / 'chunk', 'identity', 7, 4, 3,
               [('p1', 1.0, 2.0, 10, 20, first, head),
                ('p2', 3.0, 4.0, 30, 40, second, head * 2)],
               {'missing': 1, 'black': 0, 'unreadable': 0})
    archive = np.load(tmp_path / 'chunk/vectors.npz')
    assert archive['panoid'].tolist() == ['p1', 'p2']
    assert archive['composition'].shape == (2, 4096)
    assert archive['heading_groups'].shape == (2, 4, 64)
    assert archive['grid_x'].tolist() == [10, 30]
    assert archive['grid_y'].tolist() == [20, 40]
    assert float(archive['composition'][1, 0]) == 2
    marker = json.loads((tmp_path / 'chunk/complete.json').read_text())
    assert marker['processed_panoramas'] == 2 and marker['skipped']['missing'] == 1
    assert marker['selection_rows'] == 4
