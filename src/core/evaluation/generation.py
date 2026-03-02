import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def generate_continuation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[str, list[int]]:
    """Generate text continuation using the model.

    Returns (generated_text, generated_token_ids).
    Uses greedy decoding by default (temperature=0) for reproducibility.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        if temperature == 0.0:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
            )

    new_token_ids = output_ids[0, input_len:].tolist()
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return generated_text, new_token_ids


def batch_generate_continuations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    show_progress: bool = True,
) -> list[tuple[str, list[int]]]:
    """Generate continuations for a list of prompts using true batched inference.

    Pads prompts on the left (as required for causal LMs) and processes them
    in batches for much faster throughput. Uses KV cache for efficient
    autoregressive decoding.

    Returns list of (generated_text, generated_token_ids).
    """
    # Ensure tokenizer is set up for batched left-padding
    original_padding_side = tokenizer.padding_side
    original_pad_token = tokenizer.pad_token
    original_pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: list[tuple[str, list[int]] | None] = [None] * len(prompts)

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating batches", total=n_batches)

    try:
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]

            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(model.device)
            input_lengths = inputs["attention_mask"].sum(dim=1)

            with torch.no_grad():
                if temperature == 0.0:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                else:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        use_cache=True,
                    )

            for i in range(len(batch_prompts)):
                input_len = input_lengths[i].item()
                # output_ids includes padding + input + generated
                # With left-padding, the actual input starts at different offsets
                # but generate() returns sequences starting from the padded input
                total_input_len = inputs["input_ids"].shape[1]
                new_tokens = output_ids[i, total_input_len:]
                new_token_ids = new_tokens.tolist()
                # Remove any pad tokens that might appear
                if tokenizer.pad_token_id in new_token_ids:
                    new_token_ids = [
                        t for t in new_token_ids if t != tokenizer.pad_token_id
                    ]
                generated_text = tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                )
                results[start + i] = (generated_text, new_token_ids)
    finally:
        # Restore original tokenizer settings
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token = original_pad_token
        tokenizer.pad_token_id = original_pad_token_id

    return results
