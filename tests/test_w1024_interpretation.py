from formal.interpretation.build_w1024_audit import (
    old_match_group,
    prevalence_group,
    select_stratified_pilot,
)


def test_prevalence_groups_cover_all_ranges():
    assert prevalence_group(0) == "below_threshold"
    assert prevalence_group(1) == "city_unique"
    assert prevalence_group(2) == "pair_specific"
    assert prevalence_group(5) == "regional_3to5"
    assert prevalence_group(8) == "accessory_6to8"
    assert prevalence_group(11) == "near_core_9to11"
    assert prevalence_group(12) == "core_12"


def test_old_match_group_thresholds():
    assert old_match_group(0.649) == "weak"
    assert old_match_group(0.65) == "medium"
    assert old_match_group(0.75) == "strong"


def test_pilot_selection_is_unique_deterministic_and_sized():
    rows = []
    for gene in range(80):
        rows.append({
            "gene_id": gene,
            "support_all_slots": 10 + gene,
            "prevalence_group": prevalence_group(gene % 13),
            "old_match_group": ("weak", "medium", "strong")[gene % 3],
            "position_r2_top1": 0.1 if gene % 17 == 0 else 0.0,
        })
    first = select_stratified_pilot(rows, target=32, seed=42)
    second = select_stratified_pilot(rows, target=32, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 32
