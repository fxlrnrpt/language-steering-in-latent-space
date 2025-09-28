from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd

from core.synthetic_data.generate_code_switching_single_word import generate_code_switching_single_word

if __name__ == "__main__":
    freeze_support()

    SUBSET_SIZE = 1000

    df_path = Path(__file__).parent.joinpath("../../../data/wikipedia.jsonl")
    df = pd.read_json(df_path, lines=True)

    out_path = Path(__file__).parent.joinpath("../../../data/wikipedia_code_switching_single_word.jsonl")
    generate_code_switching_single_word(df, "rus_Cyrl", out_path)
