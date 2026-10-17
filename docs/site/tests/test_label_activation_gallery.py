"""Label galleries must preserve real activation evidence and panorama diversity."""
from docs.site.label_activation_gallery import select_examples


def test_select_examples_prefers_distinct_dimensions_and_panoramas():
    rows = [
        {'feature': 1, 'rank': 1, 'activation': .9, 'city': 'A', 'pano': 'first'},
        {'feature': 1, 'rank': 2, 'activation': .8, 'city': 'A', 'pano': 'second'},
        {'feature': 2, 'rank': 1, 'activation': .7, 'city': 'B', 'pano': 'first'},
        {'feature': 3, 'rank': 1, 'activation': .6, 'city': 'A', 'pano': 'first'},
        {'feature': 4, 'rank': 1, 'activation': 0, 'city': 'C', 'pano': 'unobserved'},
    ]
    selected = select_examples(rows, 3)
    assert [(row['feature'], row['pano']) for row in selected] == [
        (1, 'first'), (2, 'first'), (1, 'second')]
    assert len({(row['city'], row['pano']) for row in selected}) == 3
