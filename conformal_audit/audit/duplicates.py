"""Exact and perceptual duplicate audits."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from conformal_audit.audit.overlap import IMAGE_EXTENSIONS
from conformal_audit.config import ExperimentConfig, require_path
from conformal_audit.utils.io import write_json


def run_duplicate_audit_from_config(config: ExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.paths.get("output_dir", f"assets/output/audits/{config.name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result = audit_duplicates(
        dataset_a_name=str(config.metadata.get("dataset_a_name", "dataset_a")),
        dataset_b_name=str(config.metadata.get("dataset_b_name", "dataset_b")),
        dataset_a_dir=require_path(config.paths["dataset_a_dir"], "dataset_a_dir"),
        dataset_b_dir=require_path(config.paths["dataset_b_dir"], "dataset_b_dir"),
        hash_size=int(config.metadata.get("hash_size", 8)),
        max_images_per_dataset=_optional_int(config.metadata.get("max_images_per_dataset")),
    )
    result["name"] = config.name
    result["output_dir"] = str(output_dir)
    write_json(output_dir / "duplicate_audit_report.json", result)
    write_group_csv(output_dir / "cross_file_hash_duplicates.csv", result["cross_file_hash_groups"])
    write_group_csv(output_dir / "cross_perceptual_hash_duplicates.csv", result["cross_perceptual_hash_groups"])
    write_group_csv(output_dir / "within_dataset_file_hash_duplicates.csv", result["within_dataset_file_hash_groups"])
    write_group_csv(output_dir / "within_dataset_perceptual_hash_duplicates.csv", result["within_dataset_perceptual_hash_groups"])
    return result


def audit_duplicates(
    dataset_a_name: str,
    dataset_b_name: str,
    dataset_a_dir: str | Path,
    dataset_b_dir: str | Path,
    hash_size: int = 8,
    max_images_per_dataset: int | None = None,
) -> dict[str, Any]:
    """Find exact file-hash and exact perceptual-hash duplicate groups."""

    inventory_a = build_hash_inventory(dataset_a_dir, dataset_a_name, hash_size, max_images_per_dataset)
    inventory_b = build_hash_inventory(dataset_b_dir, dataset_b_name, hash_size, max_images_per_dataset)

    cross_file_groups = cross_hash_groups(inventory_a, inventory_b, "file_sha256")
    cross_perceptual_groups = cross_hash_groups(inventory_a, inventory_b, "average_hash")
    within_file_groups = within_hash_groups(inventory_a + inventory_b, "file_sha256")
    within_perceptual_groups = within_hash_groups(inventory_a + inventory_b, "average_hash")

    return {
        "dataset_a": dataset_a_name,
        "dataset_b": dataset_b_name,
        "dataset_a_dir": str(dataset_a_dir),
        "dataset_b_dir": str(dataset_b_dir),
        "dataset_a_image_count": len(inventory_a),
        "dataset_b_image_count": len(inventory_b),
        "hash_size": hash_size,
        "max_images_per_dataset": max_images_per_dataset,
        "failed_image_reads": failed_reads(inventory_a + inventory_b),
        "cross_file_hash_group_count": len(cross_file_groups),
        "cross_file_hash_image_count": grouped_image_count(cross_file_groups),
        "cross_perceptual_hash_group_count": len(cross_perceptual_groups),
        "cross_perceptual_hash_image_count": grouped_image_count(cross_perceptual_groups),
        "within_dataset_file_hash_group_count": len(within_file_groups),
        "within_dataset_perceptual_hash_group_count": len(within_perceptual_groups),
        "cross_file_hash_groups": cross_file_groups,
        "cross_perceptual_hash_groups": cross_perceptual_groups,
        "within_dataset_file_hash_groups": within_file_groups,
        "within_dataset_perceptual_hash_groups": within_perceptual_groups,
        "note": (
            "Perceptual-hash groups are exact average-hash matches. They are useful screening signals "
            "and should be reviewed before treating non-identical filenames as confirmed duplicates."
        ),
    }


def build_hash_inventory(
    dataset_dir: str | Path,
    dataset_name: str,
    hash_size: int = 8,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    image_paths = sorted(
        path
        for path in Path(dataset_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]

    inventory: list[dict[str, Any]] = []
    for path in image_paths:
        record = {
            "dataset": dataset_name,
            "path": str(path),
            "image_id": path.stem,
            "file_sha256": file_sha256(path),
            "average_hash": None,
            "read_error": None,
        }
        try:
            record["average_hash"] = average_hash(path, hash_size)
        except Exception as exc:
            record["read_error"] = f"{type(exc).__name__}: {exc}"
        inventory.append(record)
    return inventory


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: str | Path, hash_size: int = 8) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)
    bits = pixels >= float(np.mean(pixels))
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def cross_hash_groups(
    inventory_a: list[dict[str, Any]],
    inventory_b: list[dict[str, Any]],
    hash_key: str,
) -> list[dict[str, Any]]:
    groups_a = group_by_hash(inventory_a, hash_key)
    groups_b = group_by_hash(inventory_b, hash_key)
    rows: list[dict[str, Any]] = []
    for hash_value in sorted(set(groups_a) & set(groups_b)):
        rows.append(make_group_row(hash_key, hash_value, groups_a[hash_value], groups_b[hash_value]))
    return rows


def within_hash_groups(inventory: list[dict[str, Any]], hash_key: str) -> list[dict[str, Any]]:
    by_dataset_and_hash: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in inventory:
        hash_value = record.get(hash_key)
        if hash_value:
            by_dataset_and_hash[(str(record["dataset"]), str(hash_value))].append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, hash_value), records in sorted(by_dataset_and_hash.items()):
        if len(records) > 1:
            rows.append(
                {
                    "hash_type": hash_key,
                    "hash_value": hash_value,
                    "dataset": dataset,
                    "image_count": len(records),
                    "sample_paths": ";".join(record["path"] for record in records[:10]),
                    "sample_image_ids": ";".join(record["image_id"] for record in records[:10]),
                }
            )
    return rows


def group_by_hash(inventory: list[dict[str, Any]], hash_key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inventory:
        hash_value = record.get(hash_key)
        if hash_value:
            groups[str(hash_value)].append(record)
    return groups


def make_group_row(
    hash_key: str,
    hash_value: str,
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "hash_type": hash_key,
        "hash_value": hash_value,
        "dataset_a_count": len(records_a),
        "dataset_b_count": len(records_b),
        "dataset_a_sample_paths": ";".join(record["path"] for record in records_a[:10]),
        "dataset_b_sample_paths": ";".join(record["path"] for record in records_b[:10]),
        "dataset_a_sample_image_ids": ";".join(record["image_id"] for record in records_a[:10]),
        "dataset_b_sample_image_ids": ";".join(record["image_id"] for record in records_b[:10]),
    }


def write_group_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_image_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        total += int(row.get("dataset_a_count", 0)) + int(row.get("dataset_b_count", 0))
    return total


def failed_reads(inventory: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"dataset": str(record["dataset"]), "path": str(record["path"]), "read_error": str(record["read_error"])}
        for record in inventory
        if record.get("read_error")
    ][:100]


def _optional_int(value: object) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)
