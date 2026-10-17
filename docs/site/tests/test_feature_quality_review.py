"""Review candidates remain evidence cues, not automatic feature exclusions."""
import json

from docs.site import feature_quality_review as review


def test_priority_position_review_keeps_content_position_as_a_question():
    screening = json.loads(review.SCREENING.read_text())['dimensions']
    hierarchy = json.loads(review.HIERARCHY.read_text())['dimensions']
    priority = {i for i, (screen, label) in enumerate(zip(screening, hierarchy))
                if review.priority_position_review(screen, label['family'])}
    assert priority == {690, 1483, 1807, 3156, 3781}
    assert screening[3414]['peak_4x4_max_bin_fraction'] >= .98
    assert not review.priority_position_review(screening[3414], hierarchy[3414]['family'])
    assert all(screening[i]['position_coding_confirmed'] is False for i in priority)


def test_blur_queue_preserves_weak_labels_as_review_cues():
    hierarchy = json.loads(review.HIERARCHY.read_text())['dimensions']
    candidates = {row['feature']: review.blur_evidence(
        row['label_zh'], row['fine_label_zh'], row['feature']) for row in hierarchy}
    assert sum(value == 'old_qwen_label' for value in candidates.values()) == 29
    assert sum(value == 'visual_spot_check' for value in candidates.values()) == 4
    assert candidates[1711] == candidates[3829] == 'old_qwen_label'
    assert candidates[694] == candidates[900] == 'visual_spot_check'
    assert not candidates[2193]


def test_peak_bins_use_the_four_by_four_view_grid(tmp_path):
    source = tmp_path / 'peaks.jsonl'
    source.write_text('\n'.join(json.dumps({'feature': 7, 'peak_patch': peak})
                                for peak in [0, 6, 7, 196, 783]) + '\n')
    bins = review.peak_bins(source)[7]
    assert len(bins) == 16 and sum(bins) == 5
    assert bins[0] == 2 and bins[1] == 1 and bins[4] == 1 and bins[15] == 1
