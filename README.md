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
ollama pull gemma3:12b
```

---

## Dataset

60 images total — 20 per class — drawn from three independent HuggingFace sources to maximize within-class diversity:

| Class | Source | Filenames |
|-------|--------|-----------|
| Real | [lmms-lab/flickr30k](https://huggingface.co/datasets/lmms-lab/flickr30k) | original Flickr photo IDs (e.g. `127332812.jpg`) |
| AI-generated | [bitmind/FakeClue](https://huggingface.co/datasets/bitmind/FakeClue) | `fakeclue_000.jpg` – `fakeclue_019.jpg` |
| Deepfake | [saakshigupta/deepfake-detection-dataset-v3](https://huggingface.co/datasets/saakshigupta/deepfake-detection-dataset-v3) | `df_000.jpg` – `df_019.jpg` |

To download, add your HuggingFace token to a `.env` file and run:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
python src/download_dataset.py
```

Images are saved to `data/raw/`. Labels are written to `data/processed/labels.csv`.

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
