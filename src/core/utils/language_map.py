language_code_to_name = {
    "eng_Latn": "English",
    "spa_Latn": "Spanish",
    "cmn_Hans": "Mandarin",
    "rus_Cyrl": "Russian",
    "hin_Deva": "Hindi",
}
language_name_to_code = {v: k for k, v in language_code_to_name.items()}
language_index_to_code = [k for k in language_code_to_name.keys()]
language_index_to_name = [k for k in language_code_to_name.values()]


def check_language_code(language_code: str):
    assert language_code in language_index_to_code
