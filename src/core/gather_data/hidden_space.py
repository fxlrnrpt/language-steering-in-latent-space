from typing import List

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def collect_hidden_space_by_language(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data: List[dict[str, str]],
    mask_special_tokens=True,
    skip_first=False,
):
    """
    Returns { [lang]: torch.Tensor([n_layers, n_tokens, d_model]) }, { [lang]: [n_tokens] }
    """
    print("Data len: ", len(data))

    # { [lang]: torch.tensor([n_layers, n_tokens, d_model]) }
    hidden_space_for_language = {}
    # { [lang]: [n_tokens] }
    token_map_for_language = {}

    N = model.config.num_hidden_layers + 1

    for entry in tqdm(data):
        for language, text in entry.items():
            if language not in hidden_space_for_language:
                hidden_space_for_language[language] = torch.zeros((N, 0, model.config.hidden_size))
                token_map_for_language[language] = []

            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True, return_special_tokens_mask=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                B, T = inputs["input_ids"].shape
                assert B == 1

                out = model.forward(**inputs, output_hidden_states=True, return_dict=True)
                per_layer_token_embs = torch.stack(out.hidden_states, dim=0)
                per_layer_token_embs = per_layer_token_embs.squeeze(1)
                assert per_layer_token_embs.shape == (N, T, model.config.hidden_size)

                input_ids_list = inputs["input_ids"][0]

                # Exclude special and padded tokens
                if mask_special_tokens:
                    special_mask = inputs["special_tokens_mask"].squeeze(0).bool()  # (T,)
                    attn_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).squeeze(0).bool()
                    non_special_token_mask = (~special_mask) & attn_mask  # (T,)

                    per_layer_token_embs = per_layer_token_embs[:, non_special_token_mask, :]
                    input_ids_list = input_ids_list[non_special_token_mask]

                if skip_first:
                    per_layer_token_embs = per_layer_token_embs[:, 1:, :]
                    input_ids_list = input_ids_list[1:]

                hidden_space_for_language[language] = torch.cat(
                    [hidden_space_for_language[language], per_layer_token_embs.cpu().detach()], dim=1
                )
                token_map_for_language[language] += input_ids_list.cpu().tolist()

    return hidden_space_for_language, token_map_for_language
