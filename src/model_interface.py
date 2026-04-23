import ollama
from pathlib import Path

MODEL = "gemma3:12b"


def query(prompt: str, test_image: Path, example_images: list[Path] | None = None) -> str:
    """Send a prompt + image(s) to the model and return the raw text response.

    For few-shot runs, pass example_images in label order (real, ai_generated, deepfake).
    They are prepended to the test image in the images array.
    """
    images = [str(p) for p in (example_images or [])] + [str(test_image)]
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": images}],
    )
    return response.message.content.strip()
