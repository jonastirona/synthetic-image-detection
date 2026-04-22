"""
Downloads 20 images per class (Real, Artificial, Deepfake) from
prithivMLmods/AI-vs-Deepfake-vs-Real on HuggingFace and saves them
to data/raw/ with consistent naming.
"""

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

dataset = load_dataset("prithivMLmods/AI-vs-Deepfake-vs-Real", split="train")

counts = {label: 0 for label in CLASS_MAP}
label_names = dataset.features["label"].names

for row in dataset:
    label_str = label_names[row["label"]]
    if label_str not in counts or counts[label_str] >= N:
        continue
    prefix = CLASS_MAP[label_str]
    idx = counts[label_str] + 1
    filename = DEST / f"{prefix}_{idx:02d}.jpg"
    row["image"].convert("RGB").save(filename, "JPEG")
    counts[label_str] += 1
    print(f"Saved {filename}")
    if all(v >= N for v in counts.values()):
        break

print("\nDone.")
for label, count in counts.items():
    print(f"  {label}: {count} images")
