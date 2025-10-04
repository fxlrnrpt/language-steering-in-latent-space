from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd

from core.synthetic_data.generate_code_switching_single_word import (
    generate_code_switching_last_n_words,
    postprocess_source_code,
)
from core.utils.language_map import language_index_to_code

if __name__ == "__main__":
    freeze_support()

    SUBSET_SIZE = 1000

    df_path = Path(__file__).parent.joinpath("../../../data/wikipedia.jsonl")
    df = pd.read_json(df_path, lines=True)

    out_path = Path(__file__).parent.joinpath("../../../data/wikipedia_code_switching_single_word.jsonl")
    postprocess_source_code(df, df_path)
    translate_codes = [language_code for language_code in language_index_to_code if language_code != "eng_Latn"]
    for language_code in translate_codes:
        print(f"Translating to {language_code}...\n\n")
        generate_code_switching_last_n_words(df, language_code, out_path)
