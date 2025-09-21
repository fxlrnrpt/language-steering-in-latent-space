import matplotlib.pyplot as plt
import numpy as np


def visualize_explained_variance(explained_variance_ratios):
    fig = plt.figure(figsize=(12, 6))

    n_layers = len(explained_variance_ratios)

    for layer in [0, n_layers // 2 - 1, n_layers - 1]:  # Plot only a few layers for clarity
        plt.plot(
            np.arange(0, len(explained_variance_ratios[layer])),
            explained_variance_ratios[layer],
            marker="o",
            label=f"Layer {layer}",
        )

    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    plt.subplots_adjust(top=0.85, bottom=0.15)

    plt.show()

    fig = plt.figure(figsize=(12, 6))

    plt.plot(
        [i for i in range(n_layers)],
        [explained_variance_ratios[i][0] for i in range(n_layers)],
        marker="o",
        label="Explained variance for 1st PCA component",
    )

    plt.xlabel("Layer")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    plt.subplots_adjust(top=0.85, bottom=0.15)

    plt.show()
