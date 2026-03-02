import os
from pathlib import Path
from urllib.request import urlretrieve

import sys

import numpy as np

# Monkey-patch numpy 2.x incompatibility in fasttext before importing it.
# fasttext calls np.array(probs, copy=False) which fails in numpy 2.x.
# We patch the numpy reference inside the fasttext.FastText module.
import fasttext.FastText as _ft_module

_ft_module.np = type(sys)("_numpy_compat")
_ft_module.np.__dict__.update(np.__dict__)
_ft_module.np.array = lambda *a, **kw: np.asarray(*a, **{k: v for k, v in kw.items() if k != "copy"})

import fasttext  # noqa: E402


_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".cache"
_MODEL_PATH = _MODEL_DIR / "lid.176.bin"

# Map fasttext labels to ISO 639-1 codes
_FASTTEXT_TO_ISO = {
    "__label__en": "en",
    "__label__es": "es",
    "__label__ru": "ru",
    "__label__zh": "zh",
    "__label__hi": "hi",
}


def load_lid_model() -> fasttext.FastText:
    """Load fasttext lid.176.bin model (downloads if needed)."""
    if not _MODEL_PATH.exists():
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"Downloading fasttext LID model to {_MODEL_PATH}...")
        urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Done.")
    return fasttext.load_model(str(_MODEL_PATH))


def classify_text(text: str, model: fasttext.FastText) -> str:
    """Classify language of a text string. Returns ISO 639-1 code."""
    text = text.replace("\n", " ").strip()
    if not text:
        return "unk"
    predictions = model.predict(text, k=1)
    label = predictions[0][0]  # e.g. "__label__en"
    return _FASTTEXT_TO_ISO.get(label, label.replace("__label__", ""))


def classify_tokens_sliding_window(
    tokens: list[str],
    model: fasttext.FastText,
    window_size: int = 5,
) -> list[str]:
    """Classify each token using a sliding window of surrounding tokens.

    For each token position i, concatenate tokens[i-w:i+w+1], classify
    the concatenated string, assign that label to position i.
    Punctuation/whitespace-only tokens inherit the nearest non-punctuation label.
    """
    if len(tokens) == 0:
        return []

    half_w = window_size // 2
    raw_labels = []

    for i in range(len(tokens)):
        start = max(0, i - half_w)
        end = min(len(tokens), i + half_w + 1)
        window_text = "".join(tokens[start:end]).strip()
        if window_text and any(c.isalpha() for c in window_text):
            raw_labels.append(classify_text(window_text, model))
        else:
            raw_labels.append(None)  # punctuation/whitespace

    # Fill None labels from nearest non-None neighbor
    labels = list(raw_labels)
    for i in range(len(labels)):
        if labels[i] is not None:
            continue
        # Search left then right for nearest label
        left = right = None
        for j in range(i - 1, -1, -1):
            if labels[j] is not None:
                left = labels[j]
                break
        for j in range(i + 1, len(labels)):
            if raw_labels[j] is not None:
                right = raw_labels[j]
                break
        labels[i] = left or right or "unk"

    return labels


def classify_generated_text(
    text: str,
    tokenizer,
    model: fasttext.FastText,
    window_size: int = 5,
) -> list[tuple[str, str]]:
    """Tokenize text with the LLM tokenizer, then classify each token.

    Returns list of (token_string, language_label) pairs.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    labels = classify_tokens_sliding_window(token_strings, model, window_size)
    return list(zip(token_strings, labels))
