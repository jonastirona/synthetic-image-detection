# Synthetic Image Detection

CS 474 — Jonas Tirona

Evaluating multimodal LLM performance for detecting real, AI-generated, and deepfake images using prompting strategies.

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
ollama pull llava
```

---

## Dataset

Images are sourced from [prithivMLmods/AI-vs-Deepfake-vs-Real](https://huggingface.co/datasets/prithivMLmods/AI-vs-Deepfake-vs-Real) on HuggingFace (gated — requires free account + accepting terms).

**60 images total, 20 per class:**

| Class | Files | HuggingFace label |
|-------|-------|-------------------|
| Real | `real_01.jpg` – `real_20.jpg` | `Real` |
| AI-generated | `ai_01.jpg` – `ai_20.jpg` | `Artificial` |
| Deepfake | `deepfake_01.jpg` – `deepfake_20.jpg` | `Deepfake` |

To download, set your HuggingFace token and run:

```bash
export HF_TOKEN=hf_your_token_here
python src/download_dataset.py
```

Images are saved to `data/raw/`.

---

## Running Experiments

```bash
python src/run_experiments.py
```

480 runs: 60 images × 4 prompting strategies × 2 output formats. Results saved to `results/predictions.csv`.

---

## Evaluation

```bash
python src/evaluation.py
```

Outputs accuracy, F1, confusion matrices, and figures to `results/`.
