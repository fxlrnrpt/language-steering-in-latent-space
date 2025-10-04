from concurrent import futures
from pathlib import Path

from pandas import DataFrame
from tqdm import tqdm

from core.utils.chunker import chunker
from core.utils.language_map import check_language_code
from core.utils.openrouter import openrouter


def translate_system_prompt(target_language: str, n: int):
    return f"""
You are a professional translator.
Your task is to take the user's input text, identify the last {n} words, translate only that last {n} words into {target_language}.
Return only the input text with the last {n} words replaced by the translation.
Do not translate any other parts of the text. Preserve the original formatting and structure.
"""


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(model="google/gemini-2.5-flash", messages=messages)
        return index, completion.choices[0].message.content
    except Exception as ex:
        print(ex)
        return None


def generate_code_switching_last_n_words(
    df: DataFrame,
    target_language_code: str,
    out_filename: str | Path,
    n=1,
    chunk_size=30,
    debug_every=100,
    dump_every=100,
):
    check_language_code(target_language_code)

    source_language_code = "eng_Latn"
    check_language_code(source_language_code)

    invalid_answers = 0
    sys_prompt = translate_system_prompt(target_language_code, n)

    df[target_language_code] = ""

    with futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=int(df.shape[0] / chunk_size)):
            args_list = []

            for index, row in chunk.iterrows():
                if df.at[index, target_language_code] != "":
                    continue

                user_prompt = row[source_language_code]
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
                        df.to_json(f, orient="records", lines=True)

    with open(out_filename, "w") as f:
        df.to_json(f, orient="records", lines=True)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df


def postprocess_source_code(
    df: DataFrame,
    out_filename: str | Path,
    start_at=0,
    seq_len=30,
):
    source_language_code = "eng_Latn"
    check_language_code(source_language_code)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        df.at[index, source_language_code] = " ".join(
            row[source_language_code].split(" ")[start_at : start_at + seq_len]
        )

    with open(out_filename, "w") as f:
        df.to_json(f, orient="records", lines=True)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}.")
    return df
