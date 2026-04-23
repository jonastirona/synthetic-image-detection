# Synthetic Image Detection

CS 474 — Jonas Tirona

Evaluating multimodal LLM performance for detecting real, AI-generated, and deepfake images across four prompting strategies and two output formats.

---

## Setup

```bash
git clone https://github.com/jonastirona/synthetic-image-detection.git
cd synthetic-image-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), then pull the model:

```bash
ollama pull gemma3:12b
```

---

## Dataset

60 test images (20 per class) drawn from three independent HuggingFace sources for maximum within-class diversity. 1 held-out example image per class is stored in `data/examples/` for visual few-shot prompting.

| Class | Source | Test filenames |
|-------|--------|----------------|
| Real | [lmms-lab/flickr30k](https://huggingface.co/datasets/lmms-lab/flickr30k) | original Flickr photo IDs (e.g. `127332812.jpg`) |
| AI-generated | [bitmind/FakeClue](https://huggingface.co/datasets/bitmind/FakeClue) | `fakeclue_000.jpg` – `fakeclue_019.jpg` |
| Deepfake | [saakshigupta/deepfake-detection-dataset-v3](https://huggingface.co/datasets/saakshigupta/deepfake-detection-dataset-v3) | `df_000.jpg` – `df_019.jpg` |

Add your HuggingFace token to a `.env` file, then download:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
python src/download_dataset.py
```

Images are saved to `data/raw/`. Labels are written to `data/processed/labels.csv`.

---

## Experiment Design

**Model:** Gemma 3 12B via Ollama (local, no API)

**480 total runs:** 60 images × 4 prompting strategies × 2 output formats

| Prompting strategy | Description |
|--------------------|-------------|
| `zero_shot` | Direct question, no guidance |
| `structured` | Guided checklist of visual cues (texture, lighting, artifacts) |
| `few_shot` | 3 labeled visual examples (one per class) passed alongside the test image |
| `cot` | Step-by-step reasoning before classification |

| Output format | Description |
|---------------|-------------|
| `label_only` | Single word: `real`, `ai_generated`, or `deepfake` |
| `reasoning` | Structured response with label, confidence (0–100), and explanation |

---

## Running Experiments

```bash
python src/run_experiments.py
```

Results are written incrementally to `results/predictions.csv`. The run is resumable — already-completed rows are skipped on restart.

---

## Evaluation

```bash
python src/evaluation.py
```

Outputs accuracy, per-class F1, confusion matrices, and figures to `results/`.

---

## Results

**Overall accuracy: 41.5%** (vs. 33% random chance) across all 480 runs.

### Accuracy by Prompt Type

| Prompt | Accuracy | Macro F1 |
|--------|----------|----------|
| `structured` | 46.7% | 0.387 |
| `zero_shot` | 45.8% | 0.370 |
| `cot` | 39.2% | 0.313 |
| `few_shot` | 34.2% | 0.198 |

### Accuracy by Output Format

| Format | Accuracy | Macro F1 |
|--------|----------|----------|
| `label_only` | 44.6% | 0.373 |
| `reasoning` | 38.3% | 0.262 |

### Per-Class Performance

| Class | Recall | F1 |
|-------|--------|-----|
| `ai_generated` | 90.6% | 0.547 |
| `real` | 30.0% | 0.364 |
| `deepfake` | 3.8% | 0.072 |

### Key Findings

- The model is strongly biased toward predicting `ai_generated` (91% recall), at the cost of almost completely missing `deepfake` images (3.8% recall, precision 100% for the few it catches).
- Structured prompting performed best overall, consistent with the hypothesis that guided visual analysis improves classification.
- Contrary to expectations, requiring reasoning output (label + confidence + explanation) slightly *hurt* accuracy relative to label-only output.
- Confidence scores were poorly calibrated: correct predictions averaged 91.9 vs. incorrect predictions averaging 89.2 — nearly indistinguishable.
- Few-shot prompting underperformed zero-shot, likely because the 3 visual examples introduced class imbalance in the model's context window.
