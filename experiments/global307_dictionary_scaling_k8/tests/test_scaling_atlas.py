from pathlib import Path

import numpy as np

from formal.analysis.build_scaling_atlas import edges_for_parent, google_rows, select_images


def test_google_rows_searches_past_baidu_dominated_top_examples():
    anchors = [{"city": "Shanghai"}] * 18 + [{"city": "Marseille"}, {"city": "Oslo"}]
    values = np.arange(20, 0, -1, dtype=np.float32)
    assert google_rows(values, anchors) == [18, 19]


def test_stable_split_child_appears_once_with_both_relations():
    stable = {5: {"child": 1}}
    splits = {5: {"children": [1, 2]}}
    metrics = {
        "decoder_cosine": np.ones((6, 3)),
        "activation_cosine": np.ones((6, 3)),
        "support_jaccard": np.ones((6, 3)),
        "parent_recall": np.ones((6, 3)),
        "child_precision": np.ones((6, 3)),
    }
    assert [(edge["child"], edge["kind"]) for edge in edges_for_parent(5, stable, splits, metrics)] == [
        (1, "stable_split"), (2, "split")
    ]


def test_split_examples_use_distinct_complete_panoramas(tmp_path: Path):
    image = tmp_path / "street.jpg"
    image.touch()
    rows = [{
        "id": 6, "kind": "split",
        "candidates": [
            {"focus_child": 10, "rows": [0, 2]},
            {"focus_child": 11, "rows": [0, 2, 3, 1]},
        ],
    }]
    anchors = [{"city": "Oslo"}] * 4
    resolved = {
        0: {"city": "Oslo", "source_pano": 1, "heading": 0, "image": str(image), "images": {h: str(image) for h in (0, 90, 180, 270)}},
        1: {"city": "Oslo", "source_pano": 2, "heading": 90, "image": str(image), "images": {h: str(image) for h in (0, 90, 180, 270)}},
    }
    resolved[2] = {**resolved[0], "heading": 90}  # same pano, different heading
    resolved[3] = {**resolved[1], "source_pano": 3, "images": {0: str(image)}}
    select_images(rows, [], anchors, resolved)
    assert [example["source_pano"] for example in rows[0]["examples"]] == [1, 2]


def test_google_rows_ranks_distinct_pano_maxima():
    anchors = [{"city": "Oslo", "source_pano": 1, "heading": h} for h in (0, 0, 90)]
    anchors += [{"city": "Oslo", "source_pano": 2}, {"city": "Paris", "source_pano": 1}]
    assert google_rows(np.array([9, 8, 7, 6, 5]), anchors) == [0, 3, 4]


def test_google_rows_fast_pano_blocks():
    anchors = [dict(city=city, source_pano=pano) for city,pano in (("Oslo", 0), ("Paris", 1), ("Shanghai", 2)) for _ in range(128)]
    values = np.zeros(384, dtype=np.float32)
    values[3], values[128 + 4], values[256 + 5] = 8, 7, 100
    assert google_rows(values, anchors) == [3, 132]


def test_panorama_images_and_maps_share_heading_order(tmp_path):
    import torch
    from PIL import Image
    from formal.analysis.build_scaling_atlas import write_assets

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    paths = []
    for heading, color in zip((0, 90, 180, 270), colors):
        path = tmp_path / f"{heading}.png"
        Image.new("RGB", (10, 10), color).save(path)
        paths.append(str(path))
    source = np.repeat(np.arange(4, dtype=np.float16), 784).reshape(-1, 1)
    np.save(tmp_path / "dict_Oslo_p784.f16.npy", source)

    class Model:
        def _preactivate(self, x):
            return x

    example = {"city": "Oslo", "source_pano": 0, "heading": 180, "image_sources": paths}
    rows = [{"kind": "stable", "id": 0, "edges": [{"child": 0}], "examples": [example]}]
    write_assets(rows, [], tmp_path, {512: Model(), 1024: Model()}, torch.device("cpu"), tmp_path)
    image = Image.open(tmp_path / example["image"])
    assert image.size == (1792, 448)
    for position, expected in enumerate(colors):
        actual = image.getpixel((position * 448 + 224, 224))
        assert max(abs(a - b) for a, b in zip(actual, expected)) < 5
    mask = np.asarray(Image.open(tmp_path / example["maps"]["512:0"]))
    assert mask.shape == (28, 112, 4)
    assert list(mask[14, [14, 42, 70, 98], 3]) == sorted(mask[14, [14, 42, 70, 98], 3])
    assert mask[14, 14, 3] == 0 and mask[14, 98, 3] == 205
