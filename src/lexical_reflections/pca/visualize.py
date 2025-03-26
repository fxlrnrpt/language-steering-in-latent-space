import matplotlib.pyplot as plt
import numpy as np


def visualize_explained_variance(pca_components, explained_variance_ratios):
    plt.figure(figsize=(12, 6))

    for layer in [0, len(pca_components) // 2 - 1, len(pca_components) - 1]:  # Plot only a few layers for clarity
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


def visualize_projections(hidden_space_by_language, projections):
    languages = list(hidden_space_by_language.keys())

    n_layers = len(projections[languages[0]])

    for layer in [0, n_layers // 2 - 1, n_layers - 1]:  # Plot only a few layers for clarity
        plt.figure(figsize=(12, 6))

        for lang in languages:
            # Get projections for this language and layer
            proj = projections[lang][layer]

            # Plot the first two components
            plt.subplot(1, 2, 1)
            plt.scatter(proj[0], proj[1], label=f"{lang} (sim)", alpha=0.7, s=100)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.title(f"Words Projected onto main PCA components (Layer {layer})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.scatter(proj[-1], proj[-2], label=f"{lang} (diff)", alpha=0.7, s=100)
            plt.xlabel("Component n")
            plt.ylabel("Component n-1")
            plt.title(f"Words Projected onto last PCA components (Layer {layer})")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.show()
