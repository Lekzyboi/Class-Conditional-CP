"""Export overlap-aware BCN sensitivity inputs for the conformal-audit framework.

This script evaluates the existing ResNet-50 checkpoint on the non-overlapping
BCN images listed in `kept_images.csv`. It does not calibrate on BCN and does
not alter dataset files. The intended framework run uses ISIC calibration arrays
and these BCN-kept arrays as the sensitivity-analysis test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BCN_TO_ISIC = {
    "nevus": "nevus",
    "melanoma": "melanoma",
    "basal cell carcinoma": "basal cell carcinoma",
    "actinic keratosis": "actinic keratosis",
    "squamous cell carcinoma": "squamous cell carcinoma",
    "seborrheic keratosis": "benign keratosis",
    "vascular lesion": "vascular lesion",
    "dermatofibroma": "dermatofibroma",
}
ISIC_CLASSES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "squamous cell carcinoma",
    "vascular lesion",
]
CLASS_CODES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export overlap-aware BCN framework input arrays.")
    parser.add_argument(
        "--kept-manifest",
        default="assets/output/audits/bcn_overlap_filter_against_isic_test/kept_images.csv",
    )
    parser.add_argument("--checkpoint", default="assets/output/best_model.pth")
    parser.add_argument("--output-dir", default="assets/output/framework_inputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch, nn, Dataset, DataLoader, models, transforms, tqdm = import_torch_stack()

    kept_manifest = Path(args.kept_manifest)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not kept_manifest.exists():
        raise FileNotFoundError(f"Kept manifest does not exist: {kept_manifest}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    rows, skipped = load_kept_rows(kept_manifest)
    print(f"Kept manifest rows: {len(rows) + len(skipped)}")
    print(f"Mappable BCN rows: {len(rows)}")
    print(f"Skipped unmappable rows: {len(skipped)}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = BCNKeptDataset(rows, transform, Dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_resnet50_model(torch, nn, models, checkpoint_path, device, len(CLASS_CODES))

    probs_rows: list[np.ndarray] = []
    argmax_rows: list[int] = []
    label_rows: list[int] = []
    meta_rows: list[dict[str, str]] = []

    with torch.no_grad():
        for images, labels, batch_meta in tqdm.tqdm(loader, desc="BCN kept inference"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            probs_rows.extend(probs.cpu().numpy().astype(np.float32))
            argmax_rows.extend(torch.argmax(probs, dim=1).cpu().numpy().astype(np.int32).tolist())
            label_rows.extend(labels.numpy().astype(np.int32).tolist())
            meta_rows.extend(dict_from_collated_meta(batch_meta, len(labels)))

    probs_arr = np.stack(probs_rows, axis=0)
    labels_arr = np.asarray(label_rows, dtype=np.int32)
    argmax_arr = np.asarray(argmax_rows, dtype=np.int32)

    np.save(output_dir / "bcn_kept_probs.npy", probs_arr)
    np.save(output_dir / "bcn_kept_labels.npy", labels_arr)
    np.save(output_dir / "bcn_kept_argmax.npy", argmax_arr)
    write_predictions_csv(output_dir / "bcn_kept_predictions.csv", probs_arr, labels_arr, argmax_arr, meta_rows)
    write_json(
        output_dir / "bcn_kept_export_manifest.json",
        {
            "kept_manifest": str(kept_manifest),
            "checkpoint": str(checkpoint_path),
            "n_manifest_rows": len(rows) + len(skipped),
            "n_mappable_rows": len(rows),
            "n_skipped_unmappable_rows": len(skipped),
            "class_counts": class_counts(labels_arr),
            "skipped_diagnosis_counts": skipped_counts(skipped),
            "note": "BCN kept subset after overlap filtering; intended as sensitivity analysis, not independent validation.",
        },
    )
    print("BCN kept framework input export complete.")
    return 0


def import_torch_stack():
    try:
        import torch
        import torch.nn as nn
        import tqdm
        from torch.utils.data import Dataset, DataLoader
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        print("Missing PyTorch/torchvision runtime for BCN export.", file=sys.stderr)
        raise exc
    return torch, nn, Dataset, DataLoader, models, transforms, tqdm


class BCNKeptDataset:
    def __init__(self, rows: list[dict[str, str]], transform, dataset_base):
        self.rows = rows
        self.transform = transform
        self.dataset_base = dataset_base

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = label_for_diagnosis(row["diagnosis"])
        meta = {
            "image_id": row["image_id"],
            "path": row["path"],
            "diagnosis": row["diagnosis"],
            "lesion_id": row.get("lesion_id", ""),
            "attribution": row.get("attribution", ""),
        }
        return image, label, meta


def load_kept_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            diagnosis = row.get("diagnosis", "")
            image_path = Path(row.get("path", ""))
            if diagnosis in BCN_TO_ISIC and image_path.exists():
                rows.append(row)
            else:
                skipped.append(row)
    return rows, skipped


def load_resnet50_model(torch, nn, models, checkpoint_path: Path, device, n_classes: int):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def label_for_diagnosis(diagnosis: str) -> int:
    return ISIC_CLASSES.index(BCN_TO_ISIC[diagnosis])


def dict_from_collated_meta(batch_meta: dict[str, Any], n_rows: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx in range(n_rows):
        rows.append({key: str(values[idx]) for key, values in batch_meta.items()})
    return rows


def write_predictions_csv(
    path: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    argmax: np.ndarray,
    meta_rows: list[dict[str, str]],
) -> None:
    fieldnames = ["sample_index", "image_id", "image_path", "diagnosis", "lesion_id", "true_class", "pred_class"]
    fieldnames.extend(f"prob_{code}" for code in CLASS_CODES)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, label in enumerate(labels):
            meta = meta_rows[idx]
            row = {
                "sample_index": idx,
                "image_id": meta["image_id"],
                "image_path": meta["path"],
                "diagnosis": meta["diagnosis"],
                "lesion_id": meta.get("lesion_id", ""),
                "true_class": CLASS_CODES[int(label)],
                "pred_class": CLASS_CODES[int(argmax[idx])],
            }
            for class_idx, code in enumerate(CLASS_CODES):
                row[f"prob_{code}"] = float(probs[idx, class_idx])
            writer.writerow(row)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {code: int(np.sum(labels == idx)) for idx, code in enumerate(CLASS_CODES)}


def skipped_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        diagnosis = row.get("diagnosis", "")
        counts[diagnosis] = counts.get(diagnosis, 0) + 1
    return dict(sorted(counts.items()))


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
