"""
Regenerate test-set predictions from best_model.pth.

Replicates the exact data-loading, split, and inference logic of paper1.py
(lines 210-295). Inference only ? no training, no conformal calibration.

Outputs (assets/output/test_evaluation/raw/):
  test_softmax.npy        shape (3096, 8), float32
  test_argmax.npy         shape (3096,),   int32
  test_labels.npy         shape (3096,),   int32
  test_predictions.csv    per-sample table
  class_index_mapping.json  documents alphabetical -> training-order mapping
"""

import os
import sys
import random
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torchvision import models, transforms
from torchvision.datasets import ImageFolder  # standard torchvision, NOT datasets.py
from torch.utils.data import DataLoader, Subset

# ============================================================
# CONFIGURATION ? mirrors paper1.py verbatim
# ============================================================
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_DIR     = "assets/ISIC_2019_Test_Organized"
MODEL_PATH   = "assets/output/best_model.pth"
OUTPUT_DIR   = "assets/output/test_evaluation/raw"

# Training-order class names (defines index 0-7 for model outputs and stored labels)
CLASS_NAMES_TRAINING = [
    "actinic keratosis",        # 0  AK
    "basal cell carcinoma",     # 1  BCC
    "benign keratosis",         # 2  BKL
    "dermatofibroma",           # 3  DF
    "melanoma",                 # 4  MEL
    "nevus",                    # 5  NV
    "squamous cell carcinoma",  # 6  SCC
    "vascular lesion",          # 7  VASC
]
CLASS_CODES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
N_CLASSES   = 8

# Expected values from audit (test_set_results.json + complete_results.json)
EXPECTED_COUNTS = {
    "AK": 168, "BCC": 491, "BKL": 316, "DF": 46,
    "MEL": 679, "NV": 1272, "SCC": 87, "VASC": 37,
}
EXPECTED_ACC = {
    "AK": 0.2619, "BCC": 0.7902, "BKL": 0.4715, "DF": 0.4130,
    "MEL": 0.6730, "NV": 0.8640, "SCC": 0.3563, "VASC": 0.4865,
}
EXPECTED_OVERALL_ACC = 0.7122
TOLERANCE = 0.001

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: Load model  (mirrors paper1.py load_model_by_name)
# ============================================================
print(f"Device: {DEVICE}")
print(f"Loading ResNet-50 from {MODEL_PATH} ...")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  checkpoint epoch={ckpt.get('epoch','?')}  "
          f"val_acc={ckpt.get('acc', float('nan')):.4f}")
else:
    model.load_state_dict(ckpt)

model = model.to(DEVICE)
model.eval()

# ============================================================
# STEP 2: Build dataset and split  (mirrors paper1.py exactly)
# ============================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print(f"\nLoading ImageFolder from {TEST_DIR} ...")
full_dataset = ImageFolder(root=TEST_DIR, transform=transform)
print(f"  Total images in folder: {len(full_dataset)}")
print(f"  Alphabetical classes:   {full_dataset.classes}")

# Build class_mapping: alphabetical index -> training-order index
class_mapping = {}
for alpha_idx, name in enumerate(full_dataset.classes):
    if name in CLASS_NAMES_TRAINING:
        class_mapping[alpha_idx] = CLASS_NAMES_TRAINING.index(name)
print(f"  Class mapping (alpha->train): {class_mapping}")

# Replicate paper1.py split verbatim
random.seed(42)
torch.manual_seed(42)

all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)

cal_size    = len(full_dataset) // 2          # 3095 for N=6191
test_indices = all_indices[cal_size:]          # 3096 samples

print(f"\nSplit: cal={cal_size}  test={len(test_indices)}")

test_subset = Subset(full_dataset, test_indices)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

# ============================================================
# STEP 3: Inference
# ============================================================
print(f"\nRunning inference on {len(test_subset)} test samples ...")

softmax_rows = []
argmax_rows  = []
label_rows   = []
path_rows    = []
sample_pos   = 0  # position within test_indices

with torch.no_grad():
    for images, labels in tqdm.tqdm(test_loader, desc="Inference"):
        images  = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)  # (batch, 8)

        for i in range(len(images)):
            orig_label   = labels[i].item()
            mapped_label = class_mapping.get(orig_label, orig_label)
            pred_idx     = torch.argmax(probs[i]).item()
            img_path     = full_dataset.samples[test_indices[sample_pos]][0]

            softmax_rows.append(probs[i].cpu().numpy().astype(np.float32))
            argmax_rows.append(pred_idx)
            label_rows.append(mapped_label)
            path_rows.append(img_path)
            sample_pos += 1

assert sample_pos == len(test_indices), \
    f"Mismatch: expected {len(test_indices)} samples, processed {sample_pos}"

# ============================================================
# STEP 4: Save artifacts
# ============================================================
softmax_arr = np.stack(softmax_rows, axis=0)               # (3096, 8) float32
argmax_arr  = np.array(argmax_rows, dtype=np.int32)         # (3096,)
label_arr   = np.array(label_rows, dtype=np.int32)          # (3096,)

np.save(os.path.join(OUTPUT_DIR, "test_softmax.npy"), softmax_arr)
np.save(os.path.join(OUTPUT_DIR, "test_argmax.npy"),  argmax_arr)
np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"),  label_arr)
print(f"\nSaved test_softmax.npy  shape={softmax_arr.shape} dtype={softmax_arr.dtype}")
print(f"Saved test_argmax.npy   shape={argmax_arr.shape}  dtype={argmax_arr.dtype}")
print(f"Saved test_labels.npy   shape={label_arr.shape}   dtype={label_arr.dtype}")

# Per-sample CSV
rows = []
for i in range(len(label_arr)):
    row = {
        "sample_index": i,
        "image_path":   path_rows[i],
        "true_class":   CLASS_CODES[label_arr[i]],
        "pred_class":   CLASS_CODES[argmax_arr[i]],
    }
    for j, code in enumerate(CLASS_CODES):
        row[f"prob_{code}"] = float(softmax_arr[i, j])
    rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"Saved test_predictions.csv  ({len(df)} rows ? {len(df.columns)} cols)")

# Class-index mapping documentation
mapping_doc = {
    "alphabetical_classes":      full_dataset.classes,
    "training_order_classes":    CLASS_NAMES_TRAINING,
    "class_codes":               CLASS_CODES,
    "alphabetical_to_training":  {str(k): v for k, v in class_mapping.items()},
    "training_index_to_code":    {str(i): c for i, c in enumerate(CLASS_CODES)},
    "code_to_training_index":    {c: i for i, c in enumerate(CLASS_CODES)},
    "note": (
        "alphabetical order == training order for this dataset, "
        "so the mapping is identity (0->0, 1->1, ..., 7->7). "
        "class_mapping is applied anyway for correctness."
    ),
}
map_path = os.path.join(OUTPUT_DIR, "class_index_mapping.json")
with open(map_path, "w") as f:
    json.dump(mapping_doc, f, indent=2)
print(f"Saved class_index_mapping.json")

# ============================================================
# STEP 5: Verification ? raise on any mismatch
# ============================================================
print("\n" + "=" * 62)
print("VERIFICATION")
print("=" * 62)

errors = []

# V-A: per-class counts
print("\n[A] Per-class sample counts:")
for i, code in enumerate(CLASS_CODES):
    actual   = int(np.sum(label_arr == i))
    expected = EXPECTED_COUNTS[code]
    ok = actual == expected
    print(f"  {code:5s}: got {actual:5d}  expected {expected:5d}  {'OK' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"Count {code}: got {actual}, expected {expected}")

# V-B: per-class argmax accuracy
print("\n[B] Per-class argmax accuracy:")
for i, code in enumerate(CLASS_CODES):
    mask = label_arr == i
    if mask.sum() > 0:
        actual_acc = float(np.mean(argmax_arr[mask] == label_arr[mask]))
    else:
        actual_acc = 0.0
    expected_acc = EXPECTED_ACC[code]
    diff = abs(actual_acc - expected_acc)
    ok = diff <= TOLERANCE
    print(f"  {code:5s}: got {actual_acc:.4f}  expected {expected_acc:.4f}  "
          f"diff {diff:.4f}  {'OK' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"Accuracy {code}: got {actual_acc:.4f}, expected {expected_acc:.4f} "
                      f"(diff {diff:.4f} > tolerance {TOLERANCE})")

# V-C: overall accuracy
overall_acc = float(np.mean(argmax_arr == label_arr))
diff_overall = abs(overall_acc - EXPECTED_OVERALL_ACC)
ok_overall = diff_overall <= TOLERANCE
print(f"\n[C] Overall accuracy: got {overall_acc:.4f}  expected {EXPECTED_OVERALL_ACC:.4f}  "
      f"diff {diff_overall:.4f}  {'OK' if ok_overall else 'FAIL'}")
if not ok_overall:
    errors.append(f"Overall accuracy: got {overall_acc:.4f}, "
                  f"expected {EXPECTED_OVERALL_ACC:.4f} (diff {diff_overall:.4f})")

# Result
print("\n" + "=" * 62)
if errors:
    print("VERIFICATION FAILED ? removing outputs to prevent inconsistent state")
    for artifact in ["test_softmax.npy", "test_argmax.npy", "test_labels.npy",
                     "test_predictions.csv", "class_index_mapping.json"]:
        p = os.path.join(OUTPUT_DIR, artifact)
        if os.path.exists(p):
            os.remove(p)
    for e in errors:
        print(f"  FAIL {e}")
    sys.exit(1)
else:
    print("ALL VERIFICATIONS PASSED OK")
    print(f"\nOutputs written to: {OUTPUT_DIR}")
    for fname in ["test_softmax.npy", "test_argmax.npy", "test_labels.npy",
                  "test_predictions.csv", "class_index_mapping.json"]:
        p = os.path.join(OUTPUT_DIR, fname)
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")
