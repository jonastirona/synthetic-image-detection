"""
Downloads 20 test images per class and 1 example image per class (for few-shot prompting).

Test images (data/raw/):
  Real:         lmms-lab/flickr30k                        — diverse Flickr photography
  AI-generated: bitmind/FakeClue                          — 100k+ diverse synthetic images
  Deepfake:     saakshigupta/deepfake-detection-dataset-v3 — FaceForensics-based manipulations

Example images (data/examples/) are held out from the test set and used as
visual few-shot examples for llama3.2-vision prompting.

Uses seed=42 throughout. Old images in data/raw/ are removed before downloading.
"""

import csv
import os
import random
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

DEST = Path("data/raw")
EXAMPLES = Path("data/examples")
LABELS_PATH = Path("data/processed/labels.csv")
N = 20
SEED = 42

# ── Cleanup old images ────────────────────────────────────────────────
DEST.mkdir(parents=True, exist_ok=True)
EXAMPLES.mkdir(parents=True, exist_ok=True)
for old in DEST.glob("*.jpg"):
    old.unlink()
print("Cleared old images from data/raw/\n")


def save(image, dest: Path, filename: str) -> Path:
    path = dest / filename
    image.convert("RGB").save(path, "JPEG")
    print(f"  {path}")
    return path


labels: list[tuple[str, str]] = []

# ── Real: lmms-lab/flickr30k ──────────────────────────────────────────
print("[1/3] Real — lmms-lab/flickr30k")
ds_real = load_dataset("lmms-lab/flickr30k", split="test", token=TOKEN)
print(f"      {len(ds_real):,} images")

rng = random.Random(SEED)
# +1 so the example image is a different index from the test set
sampled = sorted(rng.sample(range(len(ds_real)), N + 1))
example_real_idx, test_real_indices = sampled[0], sampled[1:]

row = ds_real[example_real_idx]
save(row["image"], EXAMPLES, "example_real.jpg")

for idx in test_real_indices:
    row = ds_real[idx]
    filename = row.get("filename") or f"{row.get('img_id', idx)}.jpg"
    save(row["image"], DEST, filename)
    labels.append((filename, "real"))


# ── AI-generated: bitmind/FakeClue ───────────────────────────────────
print(f"\n[2/3] AI-generated — bitmind/FakeClue (streaming, take {N + 1})")
ds_ai = load_dataset("bitmind/FakeClue", split="train", token=TOKEN, streaming=True)
samples = list(ds_ai.take(N + 1))

save(samples[0]["image"], EXAMPLES, "example_ai.jpg")
for i, row in enumerate(samples[1:]):
    filename = f"fakeclue_{i:03d}.jpg"
    save(row["image"], DEST, filename)
    labels.append((filename, "ai_generated"))


# ── Deepfake: saakshigupta/deepfake-detection-dataset-v3 ─────────────
print(f"\n[3/3] Deepfake — saakshigupta/deepfake-detection-dataset-v3")
ds_df = load_dataset("saakshigupta/deepfake-detection-dataset-v3", split="train", token=TOKEN)
print(f"      {len(ds_df):,} images")

label_names = ds_df.features["label"].names  # ['fake', 'real']
fake_val = label_names.index("fake")
fake_indices = [i for i, row in enumerate(ds_df) if row["label"] == fake_val]
print(f"      {len(fake_indices)} fake images available")

sampled = sorted(random.Random(SEED).sample(fake_indices, N + 1))
save(ds_df[sampled[0]]["image"], EXAMPLES, "example_deepfake.jpg")
for i, idx in enumerate(sampled[1:]):
    filename = f"df_{i:03d}.jpg"
    save(ds_df[idx]["image"], DEST, filename)
    labels.append((filename, "deepfake"))


# ── Write labels.csv ──────────────────────────────────────────────────
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LABELS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    for filename, label in labels:
        writer.writerow([filename, label])

counts = {l: sum(1 for _, x in labels if x == l) for l in ("real", "ai_generated", "deepfake")}
print(f"\nDone — {len(labels)} test images + 3 example images")
for label, count in counts.items():
    print(f"  {label}: {count}")
print(f"\nLabels written to {LABELS_PATH}")
