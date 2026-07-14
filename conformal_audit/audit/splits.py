"""Reproducible split generation and validation."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

from conformal_audit.audit.overlap import IMAGE_EXTENSIONS, read_csv_records
from conformal_audit.config import ExperimentConfig, require_path
from conformal_audit.utils.io import write_json


DEFAULT_GROUP_COLUMNS = ("lesion_id", "patient_id")


def run_split_audit_from_config(config: ExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.paths.get("output_dir", f"assets/output/audits/{config.name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names_training = tuple(config.metadata.get("class_names_training", []))
    group_columns = tuple(config.metadata.get("group_columns", DEFAULT_GROUP_COLUMNS))
    seed = int(config.metadata.get("seed", 42))
    calibration_fraction = float(config.metadata.get("calibration_fraction", 0.5))

    result = create_reproducible_split(
        dataset_dir=require_path(config.paths["dataset_dir"], "split dataset directory"),
        metadata_csv=config.paths.get("metadata_csv"),
        class_names_training=class_names_training,
        group_columns=group_columns,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    result["name"] = config.name
    result["output_dir"] = str(output_dir)
    write_json(output_dir / "split_indices.json", result)
    write_split_summary_csv(output_dir / "split_summary.csv", result)
    return result


def create_reproducible_split(
    dataset_dir: str | Path,
    metadata_csv: str | Path | None = None,
    class_names_training: tuple[str, ...] = (),
    group_columns: tuple[str, ...] = DEFAULT_GROUP_COLUMNS,
    seed: int = 42,
    calibration_fraction: float = 0.5,
) -> dict[str, Any]:
    records = build_image_folder_index(dataset_dir, class_names_training)
    metadata_by_image = load_metadata_by_image(metadata_csv) if metadata_csv else {}
    group_column = first_available_group_column(metadata_by_image, group_columns)

    for record in records:
        metadata = metadata_by_image.get(record["image_id"], {})
        if group_column is None:
            record["group_id"] = record["image_id"]
        else:
            record["group_id"] = str(metadata.get(group_column) or record["image_id"])

    if group_column is None:
        split = split_by_index_shuffle(records, seed, calibration_fraction)
        split_mode = "image_level_reproduction"
        warning = "No requested lesion/patient group column was available; split falls back to image-level grouping."
    else:
        split = split_by_group_shuffle(records, seed, calibration_fraction)
        split_mode = f"grouped_by_{group_column}"
        warning = None

    validation = validate_grouped_split(split["calibration_records"], split["test_records"])
    return {
        "dataset_dir": str(dataset_dir),
        "metadata_csv": str(metadata_csv) if metadata_csv else None,
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "split_mode": split_mode,
        "group_column": group_column,
        "warning": warning,
        "n_samples": len(records),
        "calibration_size": len(split["calibration_records"]),
        "test_size": len(split["test_records"]),
        "group_overlap_count": validation["group_overlap_count"],
        "group_overlap_sample": validation["group_overlap_sample"],
        "calibration_class_counts": count_classes(split["calibration_records"]),
        "test_class_counts": count_classes(split["test_records"]),
        "calibration_indices": [record["index"] for record in split["calibration_records"]],
        "test_indices": [record["index"] for record in split["test_records"]],
        "calibration_images": [record["image_id"] for record in split["calibration_records"]],
        "test_images": [record["image_id"] for record in split["test_records"]],
    }


def build_image_folder_index(dataset_dir: str | Path, class_names_training: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    root = Path(dataset_dir)
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    class_name_to_training_idx = {name: idx for idx, name in enumerate(class_names_training)}
    records: list[dict[str, Any]] = []
    index = 0
    for alphabetical_idx, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            records.append(
                {
                    "index": index,
                    "path": str(image_path),
                    "image_id": image_path.stem,
                    "class_name": class_dir.name,
                    "alphabetical_class_index": alphabetical_idx,
                    "label": class_name_to_training_idx.get(class_dir.name, alphabetical_idx),
                }
            )
            index += 1
    return records


def load_metadata_by_image(metadata_csv: str | Path | None) -> dict[str, dict[str, str]]:
    records = read_csv_records(metadata_csv)
    metadata: dict[str, dict[str, str]] = {}
    for row in records:
        image_id = row.get("image") or row.get("isic_id")
        if image_id:
            metadata[Path(image_id).stem] = row
    return metadata


def first_available_group_column(
    metadata_by_image: dict[str, dict[str, str]],
    group_columns: tuple[str, ...],
) -> str | None:
    for column in group_columns:
        if any(row.get(column) for row in metadata_by_image.values()):
            return column
    return None


def split_by_index_shuffle(
    records: list[dict[str, Any]],
    seed: int,
    calibration_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    calibration_size = int(len(records) * calibration_fraction)
    calibration_indices = set(indices[:calibration_size])
    calibration_records = [records[i] for i in indices[:calibration_size]]
    test_records = [records[i] for i in indices[calibration_size:]]
    assert len(calibration_indices) == len(calibration_records)
    return {"calibration_records": calibration_records, "test_records": test_records}


def split_by_group_shuffle(
    records: list[dict[str, Any]],
    seed: int,
    calibration_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["group_id"]), []).append(record)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    target_size = int(len(records) * calibration_fraction)
    calibration_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for group_id in group_ids:
        destination = calibration_records if len(calibration_records) < target_size else test_records
        destination.extend(groups[group_id])
    return {"calibration_records": calibration_records, "test_records": test_records}


def validate_grouped_split(
    calibration_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    cal_groups = {str(record["group_id"]) for record in calibration_records}
    test_groups = {str(record["group_id"]) for record in test_records}
    overlap = sorted(cal_groups & test_groups)
    return {"group_overlap_count": len(overlap), "group_overlap_sample": overlap[:25]}


def count_classes(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record["class_name"]) for record in records)
    return dict(sorted(counts.items()))


def write_split_summary_csv(path: str | Path, result: dict[str, Any]) -> None:
    rows = [
        {"partition": "calibration", "class_name": cls, "count": count}
        for cls, count in result["calibration_class_counts"].items()
    ]
    rows.extend(
        {"partition": "test", "class_name": cls, "count": count}
        for cls, count in result["test_class_counts"].items()
    )
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["partition", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)
