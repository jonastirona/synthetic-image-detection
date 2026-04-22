"""
Downloads 20 images per class (Real, Artificial, Deepfake) from
prithivMLmods/AI-vs-Deepfake-vs-Real on HuggingFace and saves them
to data/raw/ with consistent naming.

Uses random sampling (seed=42) so images are drawn from across the
full dataset rather than sequentially from a single source.
"""

import random
from datasets import load_dataset
from pathlib import Path

DEST = Path("data/raw")
DEST.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "Real": "real",
    "Artificial": "ai",
    "Deepfake": "deepfake",
}
N = 20
SEED = 42

dataset = load_dataset("prithivMLmods/AI-vs-Deepfake-vs-Real", split="train")
label_names = dataset.features["label"].names

# Group indices by class
class_indices = {label: [] for label in CLASS_MAP}
for i, row in enumerate(dataset):
    label_str = label_names[row["label"]]
    if label_str in class_indices:
        class_indices[label_str].append(i)

# Random sample N indices per class
rng = random.Random(SEED)
selected = {}
for label, indices in class_indices.items():
    selected[label] = sorted(rng.sample(indices, N))

# Flatten to a set for fast lookup
selected_set = {idx: label for label, indices in selected.items() for idx in indices}

# Save images
counts = {label: 0 for label in CLASS_MAP}
for i, row in enumerate(dataset):
    if i not in selected_set:
        continue
    label_str = selected_set[i]
    prefix = CLASS_MAP[label_str]
    counts[label_str] += 1
    idx = counts[label_str]
    filename = DEST / f"{prefix}_{idx:02d}.jpg"
    row["image"].convert("RGB").save(filename, "JPEG")
    print(f"Saved {filename}")
    if all(v >= N for v in counts.values()):
        break

print("\nDone.")
for label, count in counts.items():
    print(f"  {label}: {count} images")
