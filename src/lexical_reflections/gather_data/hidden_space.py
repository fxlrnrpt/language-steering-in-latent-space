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
    # { [lang]: np.array([d_model, n_prompts, n_layers]) }
    hidden_space_for_language = {}

    for language, language_caches in activation_cache.items():
        # d_model, n_prompts, n_layers
        current_hidden_space_for_language = np.zeros((model.cfg.d_model, len(language_caches), model.cfg.n_layers))

        for cache_i, cache in enumerate(language_caches):
            # layer, batch, pos, d_model
            accum_resid = cache.accumulated_resid(apply_ln=True)
            current_hidden_space_for_language[:, cache_i, :] = accum_resid[1:, 0, -1, :].cpu().numpy().T

        hidden_space_for_language[language] = current_hidden_space_for_language

    return hidden_space_for_language
