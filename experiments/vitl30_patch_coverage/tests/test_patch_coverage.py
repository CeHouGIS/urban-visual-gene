"""Synthetic math tests; real dataset computation is GPU-only."""
import math

import pandas as pd
import pytest
import torch

from experiments.vitl30_patch_coverage.code.metrics import (
    CoveragePolicy, patch_statistics, coverage_objective, coverage_summary)
from experiments.vitl30_patch_coverage.code.experiment import evaluate, selection_score
from experiments.vitl30_patch_coverage.code.run import run


def test_image_pass_requires_exact_ceil_count_not_rounded_percent():
    target = torch.ones(2, 1, 28, 28)
    prediction = torch.zeros_like(target)
    prediction[0].flatten()[:627] = 1
    prediction[1].flatten()[:628] = 1
    stats = patch_statistics(prediction, target)
    assert stats['required_count'].tolist() == [628, 628]
    assert stats['image_pass'].tolist() == [False, True]


def test_no_activation_is_ineligible_and_empty_images_fail():
    target = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
    clean = patch_statistics(target, target)
    spurious = patch_statistics(torch.ones_like(target), target)
    assert clean['valid_count'].tolist() == [1, 0]
    assert clean['image_pass'].tolist() == [True, False]
    assert clean['coverage'].tolist() == [1., 0.]
    torch.testing.assert_close(clean['valid_mask'], spurious['valid_mask'])
    assert spurious['image_nmse'][0] > clean['image_nmse'][0]


def test_fixed_quality_threshold_has_inclusive_boundary_and_fp32_math():
    target = torch.tensor([[[[2.]], [[2.]]]])
    prediction = torch.tensor([[[[4.]], [[2.]]]])
    stats = patch_statistics(prediction.bfloat16(), target.bfloat16())
    assert stats['patch_nmse'].dtype == torch.float32
    assert stats['patch_nmse'].item() == .5
    assert stats['image_pass'].item()


def test_individual_shortfalls_cannot_be_offset_by_good_images():
    # Soft coverage > .8 on average, but the second image falls short.
    policy = CoveragePolicy(temperature_start=.01, temperature_end=.01)
    target = torch.ones(2, 1, 1, 10)
    prediction = target.clone()
    prediction[1, :, :, :3] = 0
    prediction.requires_grad_(True)
    maps = torch.zeros(2, 1, 1, 10)
    loss, metrics = coverage_objective(prediction, target, maps, torch.tensor(0.), 0, policy)
    stats = patch_statistics(prediction, target, policy)
    assert stats['soft_coverage'].mean() > policy.target_fraction
    expected = torch.relu(.8 - stats['soft_coverage']).square().mean()
    torch.testing.assert_close(metrics['coverage_penalty'], expected)
    assert metrics['coverage_penalty'] > 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[1, :, :, :3] < 0).all()
    assert metrics['batch_images_meeting_80pct'] == .5


def table_for_coverage():
    return pd.DataFrame([{'row_id': row, 'heading': heading,
                           'valid_patch_count': 10, 'covered_patch_count': n, 'patch_count': 10}
                          for row, counts in [(0, [10, 10, 10, 10]), (1, [7, 8, 8, 8])]
                          for heading, n in enumerate(counts)])


def test_summary_distinguishes_pooled_coverage_image_pass_and_pano_pass():
    table = table_for_coverage()
    stats = coverage_summary(table)
    assert stats['pooled_patch_coverage'] == pytest.approx(.8875)
    assert stats['image_pass_fraction'] == 7 / 8
    assert stats['panorama_all_four_pass_fraction'] == .5
    assert not stats['all_images_meet_target']
    assert stats['semantic_coverage'] is None
    empty = table.copy()
    empty.loc[0, ['valid_patch_count', 'covered_patch_count']] = 0
    assert coverage_summary(empty)['images_without_valid_patches'] == 1
    assert coverage_summary(empty)['image_pass_fraction'] == 6 / 8


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'heading', 'counts'])
def test_summary_rejects_incomplete_or_invalid_evidence(case):
    table = table_for_coverage()
    if case == 'missing':
        table = table.iloc[1:]
    elif case == 'duplicate':
        table = pd.concat([table, table.iloc[:1]])
    elif case == 'heading':
        table.loc[0, 'heading'] = 5
    else:
        table.loc[0, 'covered_patch_count'] = 11
    with pytest.raises(ValueError):
        coverage_summary(table)


def test_checkpoint_prefers_per_image_target_before_global_reconstruction():
    high_pass = dict(image_pass_fraction=.8, mean_hard_shortfall=.1,
                     mean_image_patch_coverage=.8, mean_image_nmse=.9)
    low_pass = dict(image_pass_fraction=.7, mean_hard_shortfall=.1,
                    mean_image_patch_coverage=.9, mean_image_nmse=.1)
    assert selection_score(high_pass) > selection_score(low_pass)
    better_tail = {**high_pass, 'mean_hard_shortfall': .05}
    assert selection_score(better_tail) > selection_score(high_pass)


def test_final_test_and_cpu_execution_are_rejected_before_reading_data():
    with pytest.raises(ValueError, match='sealed'):
        evaluate(split='test')
    with pytest.raises(ValueError, match='CUDA'):
        run('cpu')
