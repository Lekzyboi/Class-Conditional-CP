from scripts.export_bcn_framework_inputs import BCN_TO_ISIC, label_for_diagnosis


def test_bcn_mapping_contains_expected_labels():
    assert BCN_TO_ISIC["seborrheic keratosis"] == "benign keratosis"
    assert label_for_diagnosis("nevus") == 5
    assert label_for_diagnosis("squamous cell carcinoma") == 6
