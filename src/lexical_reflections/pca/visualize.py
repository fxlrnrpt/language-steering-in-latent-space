import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import seaborn as sns

sns.set_theme(palette="deep")
sns.set_style("whitegrid")
sns.set_context("poster")


def visualize_explained_variance(explained_variance_ratios):
    plt.figure(figsize=(12, 6))

    n_layers = len(explained_variance_ratios)

    for layer in [0, n_layers // 2 - 1, n_layers - 1]:  # Plot only a few layers for clarity
        plt.plot(
            np.arange(1, len(explained_variance_ratios[layer]) + 1),
            explained_variance_ratios[layer],
            marker="o",
            label=f"Layer {layer}",
        )

    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance (PCA)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

    plt.figure(figsize=(12, 6))

    plt.plot(
        [i for i in range(n_layers)],
        [explained_variance_ratios[i][0] for i in range(n_layers)],
        marker="o",
        label="Explained variance for 1st PCA component",
    )

    plt.xlabel("Layer")
    plt.ylabel("Explained Variance Ratio")
    plt.title("1st component explained variance by layer")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


def visualize_projections(hidden_space_by_language, token_map_by_language, projections, tokenizer, target_layers=None):
    languages = list(hidden_space_by_language.keys())

    n_layers = len(projections[languages[0]])

    tokens_by_artist = {}
    handles = []

    if target_layers is None:
        target_layers = [0, n_layers // 2 - 1, n_layers - 1]

    for layer in target_layers:  # Plot only a few layers for clarity
        plt.figure(figsize=(10, 10))

        for lang in languages:
            # Get projections for this language and layer
            proj = projections[lang][layer]

            # Plot the first two components
            h = plt.scatter(proj[0], proj[1], label=f"{lang}", alpha=0.7, s=100)
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title(f"Embeddings (Layer {layer})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            tokens_by_artist[h] = token_map_by_language[lang]
            handles.append(h)

        cursor = mplcursors.cursor(handles, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            token = tokens_by_artist[sel.artist][sel.index]
            sel.annotation.set_text(f"[{token}] {tokenizer.decode(token)}")
            sel.annotation.get_bbox_patch().set(facecolor="lightblue", alpha=0.7)
            sel.annotation.arrow_patch.set(arrowstyle="->", facecolor="black", alpha=0.5)

        plt.show()
