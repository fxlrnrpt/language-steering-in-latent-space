import os
from pathlib import Path

import matplotlib.pyplot as plt
import mplcursors
from adjustText import adjust_text


def get_annotation_text(tokenizer, token):
    return f"{tokenizer.decode(token)}"


def visualize_projections(
    hidden_space_by_language,
    token_map_by_language,
    projections,
    tokenizer,
    target_layers=None,
    show_annotations="hover",
    save_to: str | None = None,
):
    languages = list(hidden_space_by_language.keys())

    n_layers = len(projections[languages[0]])

    tokens_by_artist = {}
    handles = []

    if target_layers is None:
        target_layers = [0, n_layers // 2 - 1, n_layers - 1]

    for layer in target_layers:  # Plot only a few layers for clarity
        fig = plt.figure(figsize=(10, 10))
        texts = []
        for lang in languages:
            # Get projections for this language and layer
            proj = projections[lang][layer]

            # Plot the first two components
            h = plt.scatter(proj[:, 0], proj[:, 1], label=f"{lang}", alpha=0.7, s=100)
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.grid(True, alpha=0.3)

            if show_annotations == "all":
                for i, token in enumerate(token_map_by_language[lang]):
                    texts.append(
                        plt.text(
                            proj[i, 0],
                            proj[i, 1],
                            get_annotation_text(tokenizer, token),
                            fontsize=24,
                        )
                    )

            tokens_by_artist[h] = token_map_by_language[lang]
            handles.append(h)

        if show_annotations == "all":
            adjust_text(texts, expand=(1.5, 1.5))

        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        plt.subplots_adjust(top=0.85, bottom=0.15)

        if show_annotations == "hover":
            cursor = mplcursors.cursor(handles, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                token = tokens_by_artist[sel.artist][sel.index]
                sel.annotation.set_text(get_annotation_text(tokenizer, token))
                sel.annotation.get_bbox_patch().set(facecolor="lightblue", alpha=0.7)
                sel.annotation.arrow_patch.set(arrowstyle="->", facecolor="black", alpha=0.5)

        plt.show()

        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            fig.savefig(Path(save_to).joinpath(f"layer_{layer}.pdf"))
