"""Verify image ranking, native crop mapping and dimension-level aggregation."""
import numpy as np
import pytest
import torch

from formal.analysis.semantic_feature_umap import (
    TopImages, crop_region, dense_sparse_image, encode_panorama_batches,
    read_manifest, semantic_prototypes, save_image_crops, source_image,
)
from sae_experiments.models.base_sae import BatchTopKSAE


def test_ranking_counts_distinct_images_and_breaks_ties_deterministically():
    top = TopImages(3, 2)
    # Even if a dimension covers many patches, one image contributes once.
    images = [np.array([[4, 0, 0], [4, 0, 0]]),
              np.array([[5, 1, 0], [0, 2, 0]]),
              np.array([[5, 0, 0], [0, 3, 0]])]
    for i, acts in enumerate(images):
        top.add(i, acts.max(axis=0))
    rows = list(top.rows())
    assert [r['image_id'] for r in rows if r['feature'] == 0] == [1, 2]
    assert [r['image_id'] for r in rows if r['feature'] == 1] == [2, 1]
    assert not any(r['feature'] == 2 for r in rows)
    np.testing.assert_array_equal(top.support, [3, 2, 0])


def test_crop_selects_peak_component_and_maps_to_native_pixels():
    heat = np.zeros((28, 28))
    heat[5:7, 8:10] = 6
    heat[6, 9] = 10
    heat[22:26, 20:25] = 8  # Disconnected activity must not inflate this crop.
    bbox, info = crop_region(heat, (896, 448), fraction=.5, margin=1, min_side=3)
    assert bbox == [224, 64, 352, 128]
    assert info['threshold_components'] == 2
    assert info['component_patches'] == 4


def test_crop_at_image_edge_stays_in_bounds_and_rejects_no_activation():
    heat = np.zeros((28, 28)); heat[0, 27] = 1
    bbox, _ = crop_region(heat, (448, 448))
    assert bbox == [400, 0, 448, 48]
    with pytest.raises(ValueError):
        crop_region(np.zeros((28, 28)), (448, 448))


def test_sparse_duplicate_slots_and_padding_do_not_create_fake_activity():
    idx = np.array([[1, 1, -1], [2, 0, 1]])
    val = np.array([[2, 4, 0], [3, 0, 0]], dtype=np.float32)
    dense = dense_sparse_image(idx, val, 3)
    np.testing.assert_array_equal(dense, [[0, 4, 0], [0, 0, 3]])
    with pytest.raises(ValueError):
        dense_sparse_image(np.array([[4]]), np.array([[1]]), 3)


def test_panorama_budget_matches_encode_and_is_independent_of_gpu_chunking():
    torch.manual_seed(13)
    model = BatchTopKSAE(5, 7, 2)
    blocks = np.random.default_rng(9).normal(size=(3, 16, 5)).astype(np.float32)
    actual = encode_panorama_batches(model, blocks, torch.device('cpu'))
    expected = np.stack([model.encode(torch.from_numpy(b)).detach().numpy() for b in blocks])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    separate = np.concatenate([encode_panorama_batches(model, b[None], torch.device('cpu')) for b in blocks])
    np.testing.assert_allclose(actual, separate, rtol=1e-5, atol=1e-6)


def test_prototypes_equal_weight_crops_and_keep_unsupported_dimensions_empty():
    # Vector magnitude is not allowed to dominate the semantic mean.
    embeddings = np.array([[10, 0], [0, 1], [2, 0], [1, 0]], dtype=np.float32)
    prototypes, count, coherence = semantic_prototypes(embeddings, [0, 0, 1, 1], 3)
    np.testing.assert_allclose(prototypes[0], [2 ** -.5, 2 ** -.5])
    np.testing.assert_allclose(prototypes[1], [1, 0])
    np.testing.assert_array_equal(prototypes[2], [0, 0])
    np.testing.assert_array_equal(count, [2, 2, 0])
    assert coherence[0] < coherence[1]
    assert coherence[2] == 0


def test_duplicate_image_manifest_is_rejected(tmp_path):
    row = '{"city":"A","pano":"P","heading":0,"image":"x.jpg"}\n'
    path = tmp_path / 'images.jsonl'; path.write_text(row + row)
    with pytest.raises(ValueError, match='Duplicate'):
        read_manifest(path, 28)


def test_native_baidu_view_and_worker_crop_share_extractor_geometry(tmp_path):
    from PIL import Image
    from formal.analysis.prepare_semantic_umap_manifest import build_manifest

    image = tmp_path / 'pano.jpg'
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[25:75, 100:150, 0] = 220
    Image.fromarray(pixels).save(image)
    np.save(tmp_path / 'dict_Shanghai_p784.f16.npy', np.zeros((3136, 1024), np.float16))
    qc = {'cities': {'Shanghai': {'verified': True, 'cached_panos': 1}}}
    rows = build_manifest([{'city': 'Shanghai', 'source_pano': 0}],
                          {'Shanghai': ['P']}, {'Shanghai': {'P': {None: str(image)}}}, qc, tmp_path)
    assert [r['row_start'] for r in rows] == [0, 784, 1568, 2352]
    assert rows[2]['view_bbox'] == [100, 25, 150, 75]
    assert source_image(rows[2]).size == (50, 50)
    assert np.array(source_image(rows[2]))[:, :, 0].mean() > 200
    heat = np.zeros((28, 28), np.float32)
    heat[10:12, 10:12] = 2
    item = {'feature': 0, 'rank': 1, 'image_id': 2, 'activation': 2.0}
    crops, errors = save_image_crops((2, rows[2], [item], [heat], (.5, 1, 3), str(tmp_path)))
    assert errors == [] and len(crops) == 1
    assert crops[0]['bbox_native_view'] == crop_region(heat, (50, 50))[0]
    assert Image.open(tmp_path / crops[0]['crop']).size == (8, 8)
    with pytest.raises(ValueError, match='QC-excluded'):
        qc['cities']['Shanghai']['bad_kept_positions'] = [0]
        build_manifest([{'city': 'Shanghai', 'source_pano': 0}],
                       {'Shanghai': ['P']}, {'Shanghai': {'P': {None: str(image)}}}, qc, tmp_path)


def test_resume_does_not_mutate_input_manifest_in_output_directory(tmp_path, monkeypatch):
    import sys
    from formal.analysis import semantic_feature_umap as pipeline

    manifest = tmp_path / 'images.jsonl'
    manifest.write_text('{"city":"A","pano":"P","heading":0,"image":"x.jpg"}\n')
    original = manifest.read_bytes()
    checkpoint = tmp_path / 'model.pt'
    torch.save({'K': 1024, 'topk': 8, 'model_type': 'batch_topk'}, checkpoint)
    monkeypatch.setattr(pipeline, 'reduce_features', lambda args: None)
    monkeypatch.setattr(sys, 'argv', ['semantic_feature_umap', '--manifest', str(manifest),
                       '--checkpoint', str(checkpoint), '--dataset', 'test', '--outdir', str(tmp_path),
                       '--expected-cities', '1', '--stage', 'reduce'])
    pipeline.main()
    pipeline.main()
    assert manifest.read_bytes() == original
