from typing import List

import numpy as np
import torch
from transformer_lens import ActivationCache, HookedTransformer


def collect_activation_cache(model: HookedTransformer, data: List[dict[str, str]]):
    activation_cache: dict[str, List[ActivationCache]] = {}
    for entry in data:
        for language, text in entry.items():
            if language not in activation_cache:
                activation_cache[language] = []

            with torch.no_grad():
                tokens = model.to_tokens(text)
                logits, cache = model.run_with_cache(tokens)
                activation_cache[language].append(cache)

    return activation_cache


def collect_hidden_space_by_language(model: HookedTransformer, activation_cache: dict[str, List[ActivationCache]]):
    """
    Returns { [lang]: np.array([n_layers, n_entries, d_model]) }
    """
    # { [lang]: np.array([n_layers, n_tokens, d_model]) }
    hidden_space_for_language = {}

    for language, language_caches in activation_cache.items():
        # n_layers, n_tokens, d_model
        current_hidden_space_for_language = np.empty((model.cfg.n_layers, 0, model.cfg.d_model))

        for cache in language_caches:
            # layer, batch, pos, d_model
            accum_resid = cache.accumulated_resid(apply_ln=True)
            # 1: - skip first pre, 0 - single batch, 1: - skip first special start of sequence token
            accum_resid_np = accum_resid[1:, 0, 1:, :].cpu().numpy()

            current_hidden_space_for_language = np.concatenate(
                [current_hidden_space_for_language, accum_resid_np],
                axis=1,
            )

        hidden_space_for_language[language] = current_hidden_space_for_language

    return hidden_space_for_language
