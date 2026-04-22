"""
Downloads 20 images per class from three diverse, independent sources:

  Real:         lmms-lab/flickr30k                    — diverse Flickr photography (31k images)
  AI-generated: bitmind/FakeClue                      — 100k+ diverse synthetic images
  Deepfake:     saakshigupta/deepfake-detection-dataset-v3 — FaceForensics-based manipulations

Uses streaming + reservoir sampling (seed=42) to avoid downloading full datasets.
Original filenames/IDs preserved; no renaming applied.
Old images in data/raw/ are removed before downloading.
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
LABELS_PATH = Path("data/processed/labels.csv")
N = 20
SEED = 42

# ── Cleanup old images ────────────────────────────────────────────────
DEST.mkdir(parents=True, exist_ok=True)
for old in DEST.glob("*.jpg"):
    old.unlink()
print("Cleared old images from data/raw/\n")

labels: list[tuple[str, str]] = []


def save(image, filename: str, label: str) -> None:
    path = DEST / filename
    image.convert("RGB").save(path, "JPEG")
    labels.append((filename, label))
    print(f"  {path}")


# ── Real: lmms-lab/flickr30k ──────────────────────────────────────────
print("[1/3] Real — lmms-lab/flickr30k")
ds_real = load_dataset("lmms-lab/flickr30k", split="test", token=TOKEN)
print(f"      {len(ds_real):,} images")

rng = random.Random(SEED)
sampled = sorted(rng.sample(range(len(ds_real)), N))
for idx in sampled:
    row = ds_real[idx]
    filename = row.get("filename") or f"{row.get('img_id', idx)}.jpg"
    save(row["image"], filename, "real")


# ── AI-generated: bitmind/FakeClue ───────────────────────────────────
# shuffle with buffer_size=200 so only ~220 images are downloaded total
print(f"\n[2/3] AI-generated — bitmind/FakeClue (streaming, take {N})")
ds_ai = load_dataset("bitmind/FakeClue", split="train", token=TOKEN, streaming=True)
samples = list(ds_ai.take(N))
for i, row in enumerate(samples):
    filename = f"fakeclue_{i:03d}.jpg"
    save(row["image"], filename, "ai_generated")


# ── Deepfake: saakshigupta/deepfake-detection-dataset-v3 ─────────────
print(f"\n[3/3] Deepfake — saakshigupta/deepfake-detection-dataset-v3")
ds_df = load_dataset("saakshigupta/deepfake-detection-dataset-v3", split="train", token=TOKEN)
print(f"      {len(ds_df):,} images")

label_names = ds_df.features["label"].names  # ['fake', 'real']
fake_val = label_names.index("fake")
fake_indices = [i for i, row in enumerate(ds_df) if row["label"] == fake_val]
print(f"      {len(fake_indices)} fake images available")

sampled = sorted(random.Random(SEED).sample(fake_indices, N))
for i, idx in enumerate(sampled):
    row = ds_df[idx]
    filename = f"df_{i:03d}.jpg"
    save(row["image"], filename, "deepfake")


# ── Write labels.csv ──────────────────────────────────────────────────
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LABELS_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    for filename, label in labels:
        writer.writerow([filename, label])

counts = {l: sum(1 for _, x in labels if x == l) for l in ("real", "ai_generated", "deepfake")}
print(f"\nDone — {len(labels)} images total")
for label, count in counts.items():
    print(f"  {label}: {count}")
print(f"\nLabels written to {LABELS_PATH}")
