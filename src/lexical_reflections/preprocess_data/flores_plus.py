from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_flores_plus(language_codes: list[str], language_name_map: dict[str, str], train_size: Optional[int] = None):
    dfs = []
    for code in language_codes:
        cols_to_keep = ["id", "text"]
        if len(dfs) == 0:
            cols_to_keep.append("topic")

        df = (
            load_dataset("openlanguagedata/flores_plus", code, split="dev")
            .to_pandas()[cols_to_keep]
            .rename(columns={"text": code})
            .set_index("id")
            .sort_index()
        )
        dfs.append(df)

    df_flores_combined = pd.concat(dfs, axis=1)
    flores_train = [
        {language_name_map[code]: row[code] for code in language_codes} for _, row in df_flores_combined.iterrows()
    ]
    flores_test = []

    if train_size is not None:
        flores_train, flores_test = train_test_split(flores_train, train_size=train_size)

    return flores_train, flores_test
