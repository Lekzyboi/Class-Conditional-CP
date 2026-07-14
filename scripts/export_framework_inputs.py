"""Export framework-ready calibration/test arrays from a trained checkpoint.

This script mirrors the current `paper1.py` ISIC split and inference path, but
stops at reproducible inputs for `conformal_audit`:

- isic_calibration_probs.npy
- isic_calibration_labels.npy
- isic_test_probs.npy
- isic_test_labels.npy
- split_indices.json
- predictions CSV files for provenance

It does not train, delete, move, or rewrite any dataset files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES_TRAINING = [
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
EXPECTED_TEST_COUNTS = {
    "AK": 168,
    "BCC": 491,
    "BKL": 316,
    "DF": 46,
    "MEL": 679,
    "NV": 1272,
    "SCC": 87,
    "VASC": 37,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export conformal-audit framework input arrays.")
    parser.add_argument("--dataset-dir", default="assets/ISIC_2019_Test_Organized")
    parser.add_argument("--checkpoint", default="assets/output/best_model.pth")
    parser.add_argument("--output-dir", default="assets/output/framework_inputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--skip-test-count-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch, nn, models, transforms, ImageFolder, DataLoader, Subset, tqdm = import_torch_stack()

    dataset_dir = Path(args.dataset_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(root=str(dataset_dir), transform=transform)
    class_mapping = build_class_mapping(dataset.classes)

    cal_indices, test_indices = reproduce_split(len(dataset), args.seed)
    print(f"Images: {len(dataset)}")
    print(f"Calibration: {len(cal_indices)}")
    print(f"Test: {len(test_indices)}")

    model = load_resnet50_model(torch, nn, models, checkpoint_path, device, len(CLASS_CODES))
    cal = run_inference(
        torch,
        DataLoader,
        Subset,
        tqdm,
        model,
        dataset,
        cal_indices,
        class_mapping,
        device,
        args.batch_size,
        args.num_workers,
        "Calibration",
    )
    test = run_inference(
        torch,
        DataLoader,
        Subset,
        tqdm,
        model,
        dataset,
        test_indices,
        class_mapping,
        device,
        args.batch_size,
        args.num_workers,
        "Test",
    )

    if not args.skip_test_count_check:
        verify_test_counts(test["labels"])

    save_arrays(output_dir, cal, test, cal_indices, test_indices, dataset.classes, class_mapping, args)
    print("Framework input export complete.")
    return 0


def import_torch_stack():
    try:
        import torch
        import torch.nn as nn
        import tqdm
        from torch.utils.data import DataLoader, Subset
        from torchvision import models, transforms
        from torchvision.datasets import ImageFolder
    except ModuleNotFoundError as exc:
        print(
            "Missing PyTorch/torchvision runtime. Activate the environment from environment.yml "
            "or install torch/torchvision before running this exporter.",
            file=sys.stderr,
        )
        raise exc
    return torch, nn, models, transforms, ImageFolder, DataLoader, Subset, tqdm


def build_class_mapping(alphabetical_classes: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for alphabetical_idx, class_name in enumerate(alphabetical_classes):
        if class_name in CLASS_NAMES_TRAINING:
            mapping[alphabetical_idx] = CLASS_NAMES_TRAINING.index(class_name)
    return mapping


def reproduce_split(n_samples: int, seed: int) -> tuple[list[int], list[int]]:
    random.seed(seed)
    indices = list(range(n_samples))
    random.shuffle(indices)
    cal_size = n_samples // 2
    return indices[:cal_size], indices[cal_size:]


def load_resnet50_model(torch, nn, models, checkpoint_path: Path, device, n_classes: int):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    torch,
    DataLoader,
    Subset,
    tqdm,
    model,
    dataset,
    indices: list[int],
    class_mapping: dict[int, int],
    device,
    batch_size: int,
    num_workers: int,
    label: str,
) -> dict[str, Any]:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs_rows: list[np.ndarray] = []
    argmax_rows: list[int] = []
    label_rows: list[int] = []
    path_rows: list[str] = []
    index_rows: list[int] = []
    sample_pos = 0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, desc=label):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            for i in range(len(images)):
                dataset_index = indices[sample_pos]
                original_label = int(labels[i].item())
                mapped_label = class_mapping.get(original_label, original_label)
                probs_rows.append(probs[i].cpu().numpy().astype(np.float32))
                argmax_rows.append(int(torch.argmax(probs[i]).item()))
                label_rows.append(int(mapped_label))
                path_rows.append(dataset.samples[dataset_index][0])
                index_rows.append(int(dataset_index))
                sample_pos += 1

    return {
        "probs": np.stack(probs_rows, axis=0),
        "argmax": np.asarray(argmax_rows, dtype=np.int32),
        "labels": np.asarray(label_rows, dtype=np.int32),
        "paths": path_rows,
        "indices": np.asarray(index_rows, dtype=np.int32),
    }


def verify_test_counts(labels: np.ndarray) -> None:
    errors = []
    for class_idx, code in enumerate(CLASS_CODES):
        actual = int(np.sum(labels == class_idx))
        expected = EXPECTED_TEST_COUNTS[code]
        if actual != expected:
            errors.append(f"{code}: got {actual}, expected {expected}")
    if errors:
        raise RuntimeError("Test split count check failed: " + "; ".join(errors))


def save_arrays(
    output_dir: Path,
    cal: dict[str, Any],
    test: dict[str, Any],
    cal_indices: list[int],
    test_indices: list[int],
    alphabetical_classes: list[str],
    class_mapping: dict[int, int],
    args: argparse.Namespace,
) -> None:
    np.save(output_dir / "isic_calibration_probs.npy", cal["probs"])
    np.save(output_dir / "isic_calibration_labels.npy", cal["labels"])
    np.save(output_dir / "isic_calibration_argmax.npy", cal["argmax"])
    np.save(output_dir / "isic_test_probs.npy", test["probs"])
    np.save(output_dir / "isic_test_labels.npy", test["labels"])
    np.save(output_dir / "isic_test_argmax.npy", test["argmax"])

    write_predictions_csv(output_dir / "isic_calibration_predictions.csv", cal)
    write_predictions_csv(output_dir / "isic_test_predictions.csv", test)
    write_json(
        output_dir / "split_indices.json",
        {
            "seed": args.seed,
            "dataset_dir": args.dataset_dir,
            "checkpoint": args.checkpoint,
            "calibration_size": len(cal_indices),
            "test_size": len(test_indices),
            "calibration_indices": cal_indices,
            "test_indices": test_indices,
        },
    )
    write_json(
        output_dir / "class_index_mapping.json",
        {
            "alphabetical_classes": alphabetical_classes,
            "training_order_classes": CLASS_NAMES_TRAINING,
            "class_codes": CLASS_CODES,
            "alphabetical_to_training": {str(key): value for key, value in class_mapping.items()},
        },
    )


def write_predictions_csv(path: Path, data: dict[str, Any]) -> None:
    fieldnames = ["sample_index", "dataset_index", "image_path", "true_class", "pred_class"]
    fieldnames.extend(f"prob_{code}" for code in CLASS_CODES)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, label in enumerate(data["labels"]):
            row = {
                "sample_index": row_idx,
                "dataset_index": int(data["indices"][row_idx]),
                "image_path": data["paths"][row_idx],
                "true_class": CLASS_CODES[int(label)],
                "pred_class": CLASS_CODES[int(data["argmax"][row_idx])],
            }
            for class_idx, code in enumerate(CLASS_CODES):
                row[f"prob_{code}"] = float(data["probs"][row_idx, class_idx])
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
