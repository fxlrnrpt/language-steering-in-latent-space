from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd
from datasets import load_dataset

if __name__ == "__main__":
    freeze_support()

    SUBSET_SIZE = 1000

    df = pd.DataFrame(
        list(
            load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            .shuffle(42)
            .take(SUBSET_SIZE)
        )
    )

    df = df[["text"]].rename(columns={"text": "eng_Latn"})

    out_path = Path(__file__).parent.joinpath("../../../data/wikipedia.jsonl")
    with open(out_path, "w") as f:
        df.to_json(f, orient="records", lines=True)
