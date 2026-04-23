"""
480 inference runs: 60 images × 4 prompt types × 2 output formats.

Results are written incrementally to results/predictions.csv so the run
is resumable — already-completed (image, prompt_type, output_format)
combinations are skipped on restart.
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.output_formats import LABEL_ONLY, REASONING
from src.model_interface import query

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
LABELS_CSV = ROOT / "data/processed/labels.csv"
RAW_DIR = ROOT / "data/raw"
EXAMPLES_DIR = ROOT / "data/examples"
PREDICTIONS_CSV = ROOT / "results/predictions.csv"
PROMPTS_DIR = ROOT / "prompts"

# ── Config ────────────────────────────────────────────────────────────
PROMPT_TYPES = ["zero_shot", "structured", "few_shot", "cot"]
OUTPUT_FORMATS = {"label_only": LABEL_ONLY, "reasoning": REASONING}
VALID_LABELS = {"real", "ai_generated", "deepfake"}

EXAMPLE_IMAGES = [
    EXAMPLES_DIR / "example_real.jpg",
    EXAMPLES_DIR / "example_ai.jpg",
    EXAMPLES_DIR / "example_deepfake.jpg",
]


def load_prompts() -> dict[str, str]:
    return {
        name: (PROMPTS_DIR / f"{name}.txt").read_text().strip()
        for name in PROMPT_TYPES
    }


def load_labels() -> dict[str, str]:
    with open(LABELS_CSV) as f:
        return {row["image_path"]: row["label"] for row in csv.DictReader(f)}


def load_done() -> set[tuple[str, str, str]]:
    if not PREDICTIONS_CSV.exists():
        return set()
    with open(PREDICTIONS_CSV) as f:
        return {
            (r["image_path"], r["prompt_type"], r["output_format"])
            for r in csv.DictReader(f)
        }


def _normalize(text: str) -> str:
    """Normalize common model label variations to canonical form."""
    text = re.sub(r"\*+", "", text)   # strip markdown bold/italic
    text = text.lower().strip()
    text = re.sub(r"ai[-\s]+generated", "ai_generated", text)
    text = re.sub(r"deep[-\s]+fake", "deepfake", text)
    return text


def parse(raw: str, fmt: str) -> tuple[str | None, int | None]:
    normalized = _normalize(raw)

    if fmt == "label_only":
        # Split on whitespace and punctuation but NOT underscores
        for token in re.split(r"[\s\n,.:;\"']+", normalized):
            if token in VALID_LABELS:
                return token, None
        return None, None

    # reasoning format
    label, confidence = None, None
    m = re.search(r"label\s*:\s*([\w_]+)", normalized)
    if m:
        candidate = _normalize(m.group(1))
        if candidate in VALID_LABELS:
            label = candidate
    m = re.search(r"confidence\s*:\s*(\d+)", raw, re.IGNORECASE)
    if m:
        confidence = int(m.group(1))
    return label, confidence


def main() -> None:
    labels = load_labels()
    prompts = load_prompts()
    done = load_done()

    fieldnames = [
        "image_path", "true_label", "prompt_type", "output_format",
        "raw_response", "predicted_label", "confidence",
    ]

    write_header = not PREDICTIONS_CSV.exists()
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)

    total = len(labels) * len(PROMPT_TYPES) * len(OUTPUT_FORMATS)
    completed = len(done)
    remaining = total - completed

    label_counts = {}
    for v in labels.values():
        label_counts[v] = label_counts.get(v, 0) + 1

    print(f"Model:     llama3.2-vision")
    print(f"Images:    {len(labels)}  " + "  ".join(f"{k}={v}" for k, v in sorted(label_counts.items())))
    print(f"Runs:      {total} total | {completed} done | {remaining} remaining")
    print(f"Output:    {PREDICTIONS_CSV}")
    print("-" * 60)

    with open(PREDICTIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for image_path, true_label in labels.items():
            for prompt_type in PROMPT_TYPES:
                for fmt_name, fmt_str in OUTPUT_FORMATS.items():
                    if (image_path, prompt_type, fmt_name) in done:
                        continue

                    full_prompt = prompts[prompt_type].format(output_format=fmt_str)
                    # Ollama llama3.2-vision only supports 1 image per call,
                    # so few_shot uses text descriptions only (no visual examples)
                    examples = None

                    completed += 1
                    print(
                        f"[{completed:3d}/{total}] {image_path} | {prompt_type} | {fmt_name} ...",
                        end=" ", flush=True,
                    )

                    try:
                        raw = query(full_prompt, RAW_DIR / image_path, examples)
                    except Exception as e:
                        raw = f"ERROR: {e}"

                    predicted, confidence = parse(raw, fmt_name)
                    correct = "✓" if predicted == true_label else "✗"
                    print(f"→ {predicted} (true: {true_label}) {correct}")

                    writer.writerow({
                        "image_path": image_path,
                        "true_label": true_label,
                        "prompt_type": prompt_type,
                        "output_format": fmt_name,
                        "raw_response": raw,
                        "predicted_label": predicted,
                        "confidence": confidence,
                    })
                    f.flush()

    print(f"\nDone. Results saved to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
