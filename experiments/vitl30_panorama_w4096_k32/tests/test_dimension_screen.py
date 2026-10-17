from experiments.vitl30_panorama_w4096_k32.code.panorama_vitl30_dimension_screen import screen_one


AUDIT = {'top_crop_count': '0', 'top_crop_cities': '0', 'status': 'unobserved_in_sample'}


def test_rare_expanded_activation_is_not_a_semantic_label():
    row = screen_one(1662, 0, 1, AUDIT, [(19, 18)], ['US/NewYork'])
    assert row['status'] == 'insufficient_support'
    assert not row['label_eligible']
    assert row['top_crop_count'] == row['top_crop_cities'] == 1
    assert not row['position_review']


def test_position_concentration_only_requests_review():
    audit = {'top_crop_count': '100', 'top_crop_cities': '20', 'status': 'coherent_visual_candidate'}
    row = screen_one(4, 200, 0, audit, [(2, 3)] * 30)
    assert row['label_eligible']
    assert row['position_review']
    assert not row['position_coding_confirmed']
