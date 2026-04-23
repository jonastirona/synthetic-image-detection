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
