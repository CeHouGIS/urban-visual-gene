import json

import pytest

from sae_experiments.io_utils import metadata_for
from sae_experiments.panorama_vitl30_dimension_vlm import parse_response


def response():
    return dict(label="厢式货车", type="object", cues=["多例出现高顶车身"],
                alternatives=[], consistency="mostly_consistent", confidence="medium",
                caveats=["仅解释高激活裁剪"])


def test_parse_vlm_preserves_unicode_content_inside_markdown():
    row = response()
    assert parse_response("```json\n" + json.dumps(row, ensure_ascii=False) + "\n```") == row


def test_missing_semantic_type_does_not_count_as_interpreted():
    row = response()
    del row["type"]
    with pytest.raises(ValueError, match="incorrect keys"):
        parse_response(json.dumps(row))


def test_truncated_response_is_rejected_even_if_it_contains_json():
    with pytest.raises(ValueError, match="truncated"):
        parse_response(json.dumps(response()), truncated=True)


def test_unknown_semantic_type_is_rejected():
    row = response()
    row["type"] = "certain_ground_truth"
    with pytest.raises(ValueError, match="invalid enum"):
        parse_response(json.dumps(row))


def test_metadata_lookup_keeps_requested_ids_with_complete_coordinates(tmp_path):
    path = tmp_path / "panoids.csv"
    path.write_text("panoid,lat,lon\na,1,2\nb,,3\nc,4,5\nd,6,7\n")
    assert metadata_for({"a", "b", "d"}, path) == {"a": (1.0, 2.0), "d": (6.0, 7.0)}
