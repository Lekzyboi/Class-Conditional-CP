"""Identifier overlap audits for dataset pairs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from conformal_audit.config import ExperimentConfig, require_path
from conformal_audit.utils.io import write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_ID_COLUMNS = ("image", "isic_id", "lesion_id", "patient_id")


def audit_identifier_overlap(
    dataset_a_name: str,
    dataset_b_name: str,
    dataset_a_dir: str | Path | None = None,
    dataset_b_dir: str | Path | None = None,
    dataset_a_metadata: str | Path | None = None,
    dataset_b_metadata: str | Path | None = None,
    id_columns: tuple[str, ...] = DEFAULT_ID_COLUMNS,
) -> dict[str, Any]:
    """Compare image stems and metadata identifiers for two datasets."""

    summary_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "dataset_a": dataset_a_name,
        "dataset_b": dataset_b_name,
        "image_stem_overlap": {},
        "metadata_identifier_overlap": {},
        "same_column_overlap": {},
        "availability": {},
    }

    if dataset_a_dir and dataset_b_dir:
        a_stems = collect_image_stems(require_path(dataset_a_dir, f"{dataset_a_name} image directory"))
        b_stems = collect_image_stems(require_path(dataset_b_dir, f"{dataset_b_name} image directory"))
        overlap = sorted(a_stems & b_stems)
        details["image_stem_overlap"] = _overlap_detail(a_stems, b_stems, overlap)
        summary_rows.append(_summary_row("image_stem", len(a_stems), len(b_stems), overlap))

    a_table = read_csv_records(dataset_a_metadata) if dataset_a_metadata else []
    b_table = read_csv_records(dataset_b_metadata) if dataset_b_metadata else []
    details["availability"] = {
        "dataset_a_metadata_columns": list(a_table[0].keys()) if a_table else [],
        "dataset_b_metadata_columns": list(b_table[0].keys()) if b_table else [],
    }

    if a_table and b_table:
        a_pool = collect_identifier_pool(a_table, id_columns)
        b_pool = collect_identifier_pool(b_table, id_columns)
        overlap = sorted(a_pool & b_pool)
        details["metadata_identifier_overlap"] = _overlap_detail(a_pool, b_pool, overlap)
        summary_rows.append(_summary_row("metadata_identifier_pool", len(a_pool), len(b_pool), overlap))

        for column in id_columns:
            a_values = collect_column_values(a_table, column)
            b_values = collect_column_values(b_table, column)
            if not a_values or not b_values:
                details["same_column_overlap"][column] = {
                    "available_in_dataset_a": bool(a_values),
                    "available_in_dataset_b": bool(b_values),
                    "overlap_count": 0,
                    "sample_overlap": [],
                }
                continue
            overlap = sorted(a_values & b_values)
            details["same_column_overlap"][column] = _overlap_detail(a_values, b_values, overlap)
            summary_rows.append(_summary_row(f"metadata_column:{column}", len(a_values), len(b_values), overlap))

    details["summary_rows"] = summary_rows
    return details


def run_overlap_audit_from_config(config: ExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.paths.get("output_dir", f"assets/output/audits/{config.name}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    id_columns = tuple(config.metadata.get("id_columns", DEFAULT_ID_COLUMNS))
    result = audit_identifier_overlap(
        dataset_a_name=str(config.metadata.get("dataset_a_name", "dataset_a")),
        dataset_b_name=str(config.metadata.get("dataset_b_name", "dataset_b")),
        dataset_a_dir=config.paths.get("dataset_a_dir"),
        dataset_b_dir=config.paths.get("dataset_b_dir"),
        dataset_a_metadata=config.paths.get("dataset_a_metadata"),
        dataset_b_metadata=config.paths.get("dataset_b_metadata"),
        id_columns=id_columns,
    )
    result["output_dir"] = str(output_dir)
    write_json(output_dir / "identifier_overlap_report.json", result)
    write_summary_csv(output_dir / "identifier_overlap_summary.csv", result["summary_rows"])
    return result


def run_overlap_filter_from_config(config: ExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.paths.get("output_dir", f"assets/output/audits/{config.name}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    id_columns = tuple(config.metadata.get("id_columns", DEFAULT_ID_COLUMNS))
    result = build_overlap_filter_manifest(
        source_name=str(config.metadata.get("source_name", "source")),
        target_name=str(config.metadata.get("target_name", "target")),
        source_dir=config.paths.get("source_dir"),
        target_dir=require_path(config.paths["target_dir"], "target image directory"),
        source_metadata=config.paths.get("source_metadata"),
        target_metadata=config.paths.get("target_metadata"),
        id_columns=id_columns,
    )
    result["name"] = config.name
    result["output_dir"] = str(output_dir)
    write_json(output_dir / "overlap_filter_summary.json", _summary_without_manifest(result))
    write_filter_manifest_csv(output_dir / "overlap_filter_manifest.csv", result["manifest_rows"])
    write_filter_manifest_csv(output_dir / "excluded_images.csv", [row for row in result["manifest_rows"] if row["exclude"]])
    write_filter_manifest_csv(output_dir / "kept_images.csv", [row for row in result["manifest_rows"] if not row["exclude"]])
    return result


def build_overlap_filter_manifest(
    source_name: str,
    target_name: str,
    target_dir: str | Path,
    source_dir: str | Path | None = None,
    source_metadata: str | Path | None = None,
    target_metadata: str | Path | None = None,
    id_columns: tuple[str, ...] = DEFAULT_ID_COLUMNS,
) -> dict[str, Any]:
    """Create a keep/exclude manifest for a target dataset using source overlap signals."""

    source_image_stems = collect_image_stems(source_dir) if source_dir else set()
    source_records = read_csv_records(source_metadata) if source_metadata else []
    target_records = read_csv_records(target_metadata) if target_metadata else []
    source_identifier_pool = collect_identifier_pool(source_records, id_columns)
    target_metadata_by_image = records_by_image_id(target_records)

    target_paths = sorted(
        path
        for path in Path(target_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    manifest_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for path in target_paths:
        image_id = path.stem
        metadata = target_metadata_by_image.get(image_id, {})
        reasons: list[str] = []

        if image_id in source_image_stems:
            reasons.append("image_stem_in_source_dir")
        if image_id in source_identifier_pool:
            reasons.append("image_stem_in_source_metadata")

        metadata_overlap_columns: list[str] = []
        for column in id_columns:
            value = metadata.get(column)
            if value is None:
                continue
            normalized = normalize_identifier(value)
            if normalized and normalized in source_identifier_pool:
                metadata_overlap_columns.append(column)
        if metadata_overlap_columns:
            reasons.append("target_metadata_identifier_in_source_metadata")

        for reason in set(reasons):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        manifest_rows.append(
            {
                "target_dataset": target_name,
                "image_id": image_id,
                "path": str(path),
                "exclude": bool(reasons),
                "reasons": ";".join(sorted(set(reasons))),
                "metadata_overlap_columns": ";".join(sorted(set(metadata_overlap_columns))),
                "diagnosis": metadata.get("diagnosis", ""),
                "lesion_id": metadata.get("lesion_id", ""),
                "attribution": metadata.get("attribution", ""),
            }
        )

    excluded = [row for row in manifest_rows if row["exclude"]]
    kept = [row for row in manifest_rows if not row["exclude"]]
    return {
        "source_name": source_name,
        "target_name": target_name,
        "source_dir": str(source_dir) if source_dir else None,
        "target_dir": str(target_dir),
        "source_metadata": str(source_metadata) if source_metadata else None,
        "target_metadata": str(target_metadata) if target_metadata else None,
        "id_columns": list(id_columns),
        "source_image_stem_count": len(source_image_stems),
        "source_identifier_count": len(source_identifier_pool),
        "target_image_count": len(manifest_rows),
        "excluded_count": len(excluded),
        "kept_count": len(kept),
        "excluded_fraction": len(excluded) / len(manifest_rows) if manifest_rows else 0.0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "manifest_rows": manifest_rows,
        "note": (
            "This manifest is non-destructive. It only marks target images that should be excluded "
            "from an independence claim because their image stem or metadata identifiers overlap the source."
        ),
    }


def collect_image_stems(root: str | Path) -> set[str]:
    root_path = Path(root)
    return {
        path.stem
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def read_csv_records(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_identifier_pool(records: list[dict[str, str]], id_columns: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for column in id_columns:
        values.update(collect_column_values(records, column))
    return values


def collect_column_values(records: list[dict[str, str]], column: str) -> set[str]:
    values: set[str] = set()
    for row in records:
        value = row.get(column)
        if value is not None:
            normalized = normalize_identifier(value)
            if normalized:
                values.add(normalized)
    return values


def records_by_image_id(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in records:
        image_id = row.get("image") or row.get("isic_id")
        if image_id:
            out[normalize_identifier(image_id)] = row
    return out


def normalize_identifier(value: str) -> str:
    return Path(str(value).strip()).stem


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_filter_manifest_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_without_manifest(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "manifest_rows"}


def _overlap_detail(values_a: set[str], values_b: set[str], overlap: list[str]) -> dict[str, Any]:
    return {
        "dataset_a_unique": len(values_a),
        "dataset_b_unique": len(values_b),
        "overlap_count": len(overlap),
        "sample_overlap": overlap[:25],
    }


def _summary_row(source: str, count_a: int, count_b: int, overlap: list[str]) -> dict[str, Any]:
    return {
        "source": source,
        "dataset_a_unique": count_a,
        "dataset_b_unique": count_b,
        "overlap_count": len(overlap),
        "sample_overlap": ";".join(overlap[:10]),
    }
