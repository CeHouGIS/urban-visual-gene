from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, box

from render_google_coverage import footprint, write_report
from render_google_road_coverage import classify_geometry, split_line


def test_road_is_split_at_vertices_and_at_most_50_metres():
    sections = list(split_line(LineString([(0, 0), (80, 0), (80, 30)])))
    lengths = [np.linalg.norm(end - start) for start, end in sections]
    assert len(sections) == 3
    assert all(length <= 50 for length in lengths)
    assert sum(lengths) == 110


def test_road_sections_use_nearest_geolocated_panorama():
    image = Image.new("RGB", (120, 20), "white")
    overlay = Image.new("RGBA", image.size)
    totals = classify_geometry(
        [LineString([(0, 0), (100, 0)])],
        type("Identity", (), {"transform": staticmethod(lambda x, y: (x, y))})(),
        box(0, -5, 100, 5),
        cKDTree([[0, 0]]),
        ImageDraw.Draw(image),
        ImageDraw.Draw(overlay),
        lambda point: (int(point[0]), 10),
    )
    assert totals == (1, 1, 1, 50.0, 50.0)


def test_image_coverage_is_clipped_to_supplied_polygon():
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    covered, extent, cell_m, area = footprint(
        np.array([50.0, 250.0]), np.array([50.0, 250.0]),
        study_boundary=boundary, radius_m=50)
    assert extent == boundary.bounds
    assert cell_m == 10
    assert area <= boundary.area / 1_000_000
    assert covered.any()


def test_empty_in_boundary_points_produce_an_empty_clipped_coverage_mask():
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    covered, extent, _, area = footprint(
        np.array([]), np.array([]), study_boundary=boundary, radius_m=50)
    assert extent == boundary.bounds
    assert not covered.any()
    assert area == 0


def test_gallery_copies_road_map_and_opens_returnable_viewer(tmp_path: Path):
    road_root = tmp_path / "roads"
    road_map = road_root / "maps" / "Country__SãoPaulo.png"
    road_map.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "green").save(road_map)
    report = road_root / "progress.csv"
    report.write_text(
        "city,status,image,covered_road_percent,covered_road_km,uncovered_road_km\n"
        "Country/SãoPaulo,mapped,maps/Country__SãoPaulo.png,40,2,3\n",
        encoding="utf-8",
    )
    output = tmp_path / "site"
    output.mkdir()
    write_report([{
        "city": "Country/SãoPaulo", "status": "mapped", "image": "maps/Country__SãoPaulo.png",
        "located_panoids": "10", "disk_panoids_any_image": "10",
        "located_to_disk_ratio": "1", "coverage_km2_estimate": "1.2",
    }], output, report)
    page = (output / "index.html").read_text(encoding="utf-8")
    assert (output / "road-maps" / road_map.name).is_file()
    assert (output / "city_road_coverage.csv").is_file()
    assert "road-maps/Country__S%C3%A3oPaulo.png" in page
    assert "返回图集" in page
    assert "viewer.showModal()" in page
