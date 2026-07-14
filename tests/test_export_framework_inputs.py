from scripts.export_framework_inputs import build_class_mapping, reproduce_split


def test_reproduce_split_matches_expected_sizes():
    cal_indices, test_indices = reproduce_split(6191, 42)
    assert len(cal_indices) == 3095
    assert len(test_indices) == 3096
    assert set(cal_indices).isdisjoint(test_indices)


def test_build_class_mapping_uses_training_order():
    classes = ["actinic keratosis", "basal cell carcinoma", "nevus"]
    mapping = build_class_mapping(classes)
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert mapping[2] == 5
