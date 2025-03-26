from typing import List

import torch
from tqdm import tqdm
from transformer_lens import ActivationCache, HookedTransformer


def collect_activation_cache(model: HookedTransformer, data: List[dict[str, str]]):
    print("Data len: ", len(data))

    activation_cache: dict[str, List[ActivationCache]] = {}
    for entry in tqdm(data):
        for language, text in entry.items():
            if language not in activation_cache:
                activation_cache[language] = []

            with torch.no_grad():
                tokens = model.to_tokens(text)
                logits, cache = model.run_with_cache(tokens)
                activation_cache[language].append(cache)

    return activation_cache
