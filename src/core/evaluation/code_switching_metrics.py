from collections import Counter
from itertools import groupby


def compute_csi(language_labels: list[str], target_lang: str) -> float:
    """Fraction of tokens not in target language."""
    if len(language_labels) == 0:
        return 0.0
    n_foreign = sum(1 for label in language_labels if label != target_lang)
    return n_foreign / len(language_labels)


def compute_m_index(language_labels: list[str]) -> float:
    """Multilingual Index: 0=monolingual, 1=maximally mixed."""
    if len(language_labels) == 0:
        return 0.0
    counts = Counter(language_labels)
    k = len(counts)
    if k <= 1:
        return 0.0
    n = len(language_labels)
    sum_pj_sq = sum((c / n) ** 2 for c in counts.values())
    return (1 - sum_pj_sq) / ((k - 1) * sum_pj_sq)


def compute_i_index(language_labels: list[str]) -> float:
    """Integration Index: ratio of switch points to token boundaries."""
    if len(language_labels) <= 1:
        return 0.0
    n_switches = sum(1 for i in range(1, len(language_labels)) if language_labels[i] != language_labels[i - 1])
    return n_switches / (len(language_labels) - 1)


def compute_language_spans(language_labels: list[str]) -> dict:
    """Mean/median/max span length per language and overall."""
    if len(language_labels) == 0:
        return {"mean": 0.0, "max": 0, "spans": []}
    spans = [sum(1 for _ in group) for _, group in groupby(language_labels)]
    return {
        "mean": sum(spans) / len(spans),
        "max": max(spans),
        "spans": spans,
    }


def compute_tlc(language_labels: list[str], target_lang: str) -> float:
    """Target Language Consistency: fraction of tokens in target lang."""
    return 1.0 - compute_csi(language_labels, target_lang)


def compute_all_metrics(language_labels: list[str], target_lang: str) -> dict:
    """Compute all code-switching metrics at once."""
    spans_info = compute_language_spans(language_labels)
    return {
        "csi": compute_csi(language_labels, target_lang),
        "m_index": compute_m_index(language_labels),
        "i_index": compute_i_index(language_labels),
        "mean_span_length": spans_info["mean"],
        "max_span_length": spans_info["max"],
        "tlc": compute_tlc(language_labels, target_lang),
    }
