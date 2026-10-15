import numpy as np
import torch

from sae_experiments.models.panorama_context import mix_panorama_context
from sae_experiments.panorama_vitl30 import EPOCHS, PATCHES, PATCHES_PER_EPOCH, orders_for_epoch
from sae_experiments.panorama_vitl30_analysis import top_candidates


def test_four_direction_ring_joins_last_to_first_and_adjacent_views():
    features = torch.zeros(1, 4, 9, 4)
    features[0, 3, 5, 0] = 1.0  # last column of heading 270, middle row
    features[0, 0, 3, 1] = 1.0  # first column of heading 0, middle row
    mixed = mix_panorama_context(features, grid_size=3, context_weight=1.0)
    assert mixed[0, 0, 3, 0] > 0
    assert mixed[0, 3, 5, 1] > 0


def test_training_schedule_visits_every_patch_and_panorama():
    assert PATCHES_PER_EPOCH * EPOCHS == PATCHES
    windows = np.concatenate([np.arange(epoch * PATCHES_PER_EPOCH,
                                        (epoch + 1) * PATCHES_PER_EPOCH)
                              for epoch in range(EPOCHS)])
    np.testing.assert_array_equal(windows, np.arange(PATCHES))
    cities = [("X/Y", None, np.arange(7), None, None)]
    order = orders_for_epoch(cities, seed=42, epoch=0, steps=7)[0]
    np.testing.assert_array_equal(np.sort(order), np.arange(7))


def test_high_activation_example_retains_heading_and_patch():
    values = np.zeros(4 * PATCHES, dtype=np.float32)
    values[PATCHES + 27] = 0.9
    latents = np.zeros_like(values, dtype=np.int32)
    groups = np.zeros(4096, dtype=np.int16)
    groups[0] = 7
    examples = top_candidates(values, latents, groups, "A/B", 5, "panoid")
    assert len(examples) == 1
    assert examples[0]["heading"] == 90
    assert examples[0]["patch"] == 27
    assert examples[0]["group"] == 7
