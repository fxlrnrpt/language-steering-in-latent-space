from concurrent import futures
from pathlib import Path

from pandas import DataFrame
from tqdm import tqdm

from core.utils.chunker import chunker
from core.utils.language_map import check_language_code
from core.utils.openrouter import openrouter


def translate_system_prompt(target_language: str):
    return f"""
You are a professional translator.
Your task is to take the user's input text, identify the last word, translate only that last word into {target_language}.
Return only the input text with the last word replaced by its translation.
Do not translate any other parts of the text. Preserve the original formatting and structure.
"""


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index, model, max_tokens = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return index, completion.choices[0].message.content
    except:
        return None


def generate_code_switching_single_word(
    df: DataFrame,
    target_language_code: str,
    out_filename: str | Path,
    seq_len=10,
    chunk_size=30,
    debug_every=100,
    dump_every=100,
):
    check_language_code(target_language_code)

    source_language_code = "eng_Latn"
    check_language_code(source_language_code)

    invalid_answers = 0
    sys_prompt = translate_system_prompt(target_language_code)

    with futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=int(df.shape[0] / chunk_size)):
            args_list = []

            for index, row in chunk.iterrows():
                if df.at[index, target_language_code] != "":
                    continue

                user_prompt = " ".join(row[source_language_code].split(" ")[:seq_len])
                args_list.append((sys_prompt, user_prompt, index))

            results = list(pool.map(call_remote_llm, args_list))

            for result in results:
                if result is None:
                    invalid_answers += 1
                    continue

                index, response = result

                df.at[index, target_language_code] = response

                if index % debug_every == 0:
                    print(f"response: {response}\n\n")

                if chunk_idx % dump_every == 0:
                    with open(out_filename, "w") as f:
                        df.to_json(f, orient="record", lines=True)

    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
