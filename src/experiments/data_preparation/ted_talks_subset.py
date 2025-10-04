from multiprocessing import freeze_support
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

if __name__ == "__main__":
    freeze_support()

    SUBSET_SIZE = 1000

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rounakbanik/ted-talks",
        "transcripts.csv",
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    df = df[["transcript"]].rename(columns={"transcript": "eng_Latn"})

    out_path = Path(__file__).parent.joinpath("../../../data/ted_talks.jsonl")
    with open(out_path, "w") as f:
        df.to_json(f, orient="records", lines=True)
