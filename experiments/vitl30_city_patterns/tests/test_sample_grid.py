from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

from experiments.vitl30_city_patterns.code import sample_grid
from experiments.vitl30_city_patterns.code.extract import candidate_chunks


def test_utm_grid_boundaries_and_hemisphere():
    assert sample_grid.utm_epsg(53.5, 10.0) == 32632
    assert sample_grid.utm_epsg(-23.5, -46.6) == 32723
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
    left = transformer.transform(499999.5, 5999999.5, direction='INVERSE')
    right = transformer.transform(500000.5, 6000000.5, direction='INVERSE')
    assert sample_grid.cell_xy(transformer, left[1], left[0]) == (499, 5999)
    assert sample_grid.cell_xy(transformer, right[1], right[0]) == (500, 6000)


def test_stable_top_ten_with_duplicate_panoids():
    rows = [{'panoid': f'p{i}', 'lat': 1., 'lon': 2.} for i in range(25)]
    selected = []
    for order in (rows + rows[:3], list(reversed(rows)) + rows[:3]):
        heaps, counts = {}, Counter()
        for row in order:
            sample_grid.offer(heaps, counts, (4, 5), row)
        selected.append({item[1] for item in heaps[(4, 5)]})
        assert len(heaps[(4, 5)]) == 10
    expected = {row['panoid'] for row in sorted(rows, key=lambda row: sample_grid.score_panoid(row['panoid']))[:10]}
    assert selected[0] == selected[1] == expected


def test_city_selection_and_extraction_chunks(tmp_path, monkeypatch):
    metadata = tmp_path / 'metadata'
    source = metadata / 'Germany/Hamburg/panoids.parquet'
    source.parent.mkdir(parents=True)
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
    positions = [transformer.transform(500100 + i * 3, 5900100, direction='INVERSE')
                 for i in range(13)]
    positions += [transformer.transform(501100 + i * 3, 5900100, direction='INVERSE')
                  for i in range(4)]
    table = pa.table({'panoid': [f'p{i}' for i in range(17)],
                      'lat': [lat for _, lat in positions],
                      'lon': [lon for lon, _ in positions],
                      'marker': [True] * 17,
                      '#images': [4] * 16 + [3]})
    pq.write_table(table, source)
    monkeypatch.setattr(sample_grid, 'METADATA', metadata)
    selection = tmp_path / 'selection'
    report = sample_grid.sample_city('Germany/Hamburg', selection)
    assert report['grid_cells'] == 2
    assert report['selected_panoramas'] == 13
    assert report['underfilled_cells'] == 1
    chunks = list(candidate_chunks('Germany/Hamburg', selection, 5))
    assert [len(rows) for _, rows, _ in chunks] == [5, 5, 3]
    assert all('grid_x' in row and 'grid_y' in row for _, rows, _ in chunks for row in rows)
    assert len({row['panoid'] for _, rows, _ in chunks for row in rows}) == 13
