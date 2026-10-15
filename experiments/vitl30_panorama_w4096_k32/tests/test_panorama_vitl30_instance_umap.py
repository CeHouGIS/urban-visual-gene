"""Instance selection must retain actual crop-embedding row alignment."""
import json

import pytest

from sae_experiments.panorama_vitl30_instance_umap import select_instances


def row(feature, rank, pano, activation=1.0):
    return {'feature': feature, 'rank': rank, 'city': 'Test/City', 'pano': pano,
            'heading': 90, 'image_id': rank, 'crop': f'crop-{rank}.jpg',
            'activation': activation}


def manifest(tmp_path, rows):
    (tmp_path / 'crops.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in rows))
    return tmp_path


def test_select_top_ranks_preserves_embedding_row_and_distinct_panoramas(tmp_path):
    source = manifest(tmp_path, [row(0, 1, 'a'), row(0, 2, 'b'), row(0, 3, 'c'),
                                  row(1, 1, 'd'), row(1, 2, 'e')])
    selected, count, total = select_instances(source, top=2, width=3)
    assert total == 5
    assert count.tolist() == [2, 2, 0]
    assert [(r['feature'], r['rank'], r['embedding_row']) for r in selected] == [
        (0, 1, 0), (0, 2, 1), (1, 1, 3), (1, 2, 4)]


@pytest.mark.parametrize('rows', [
    [row(0, 1, 'a'), row(0, 2, 'a')],
    [row(0, 1, 'a'), row(0, 3, 'b')],
    [row(0, 1, 'a', activation=0)],
])
def test_rejects_invalid_positive_ranking(tmp_path, rows):
    with pytest.raises(ValueError):
        select_instances(manifest(tmp_path, rows), top=3, width=2)
