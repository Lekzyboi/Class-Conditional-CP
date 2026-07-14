from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from conformal_audit.audit.duplicates import audit_duplicates
from conformal_audit.audit.overlap import audit_identifier_overlap, build_overlap_filter_manifest
from conformal_audit.audit.splits import create_reproducible_split


def test_overlap_audit_detects_image_stem_overlap():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = root / "a"
        b = root / "b"
        a.mkdir()
        b.mkdir()
        (a / "ISIC_1.jpg").write_text("", encoding="utf-8")
        (a / "ISIC_2.jpg").write_text("", encoding="utf-8")
        (b / "ISIC_2.jpg").write_text("", encoding="utf-8")
        result = audit_identifier_overlap("a", "b", dataset_a_dir=a, dataset_b_dir=b)
        assert result["image_stem_overlap"]["overlap_count"] == 1


def test_reproducible_split_falls_back_to_image_level_when_no_group_column():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for class_name in ["a", "b"]:
            class_dir = root / class_name
            class_dir.mkdir()
            for idx in range(2):
                (class_dir / f"{class_name}_{idx}.jpg").write_text("", encoding="utf-8")
        result = create_reproducible_split(root, seed=42, calibration_fraction=0.5)
        assert result["calibration_size"] == 2
        assert result["test_size"] == 2
        assert result["group_overlap_count"] == 0
        assert result["split_mode"] == "image_level_reproduction"


def test_duplicate_audit_detects_cross_dataset_file_hash_match():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        a = root / "a"
        b = root / "b"
        a.mkdir()
        b.mkdir()
        image = Image.new("RGB", (12, 12), color=(100, 50, 25))
        image.save(a / "same_a.jpg")
        image.save(b / "same_b.jpg")
        result = audit_duplicates("a", "b", a, b)
        assert result["cross_file_hash_group_count"] == 1
        assert result["cross_perceptual_hash_group_count"] == 1


def test_overlap_filter_marks_target_overlap_without_deleting():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        target = root / "target"
        source.mkdir()
        target.mkdir()
        (source / "ISIC_1.jpg").write_text("", encoding="utf-8")
        (target / "ISIC_1.jpg").write_text("", encoding="utf-8")
        (target / "ISIC_2.jpg").write_text("", encoding="utf-8")
        result = build_overlap_filter_manifest("source", "target", target, source_dir=source)
        assert result["excluded_count"] == 1
        assert result["kept_count"] == 1
