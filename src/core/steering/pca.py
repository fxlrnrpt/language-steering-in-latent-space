from typing import Optional

import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA


class PCASteering:
    def __init__(self, n_components=10) -> None:
        self.n_components: int = n_components
        # [n_layers, n_components, d_model]
        self.pca_components: Optional[ndarray] = None
        # [n_layers, d_model]
        self.pca_means: Optional[ndarray] = None
        # [n_layers, n_components]
        self.explained_variance_ratios: Optional[ndarray] = None

    def fit(self, hidden_space_by_language):
        """
        :param hidden_space_by_language: { [lang]: np.array([n_layers, n_tokens, d_model]) }
        """
        # n_layers, n_tokens (across all langs), d_model
        combined_embeddings = np.concat(list(hidden_space_by_language.values()), axis=1)
        n_layers = combined_embeddings.shape[0]

        pca_components = []
        pca_means = []
        explained_variance_ratios = []

        for layer in range(n_layers):
            # Apply PCA to find common directions
            pca = PCA(n_components=self.n_components)
            pca.fit_transform(combined_embeddings[layer])

            pca_components.append(pca.components_)  # Principal components [n_components, d_model]
            pca_means.append(pca.mean_)
            explained_variance_ratios.append(pca.explained_variance_ratio_)

        self.pca_components = np.array(pca_components)
        self.pca_means = np.array(pca_means)
        self.explained_variance_ratios = np.array(explained_variance_ratios)
        return self

    def transform(self, hidden_space_by_language, n_components=3):
        """
        Project each language's embeddings onto the common subspace.

        Parameters:

        - hidden_space_by_language: { [lang]: np.array([n_layers, n_tokens, d_model]) }

        Returns:
        { [lang]: np.array([n_layers, n_tokens, n_components]) }
        """
        assert self.pca_components is not None
        assert self.pca_means is not None

        n_layers = len(self.pca_components)
        for lang_embeddings in hidden_space_by_language.values():
            assert n_layers == lang_embeddings.shape[0]

        projections = {}

        for lang in hidden_space_by_language:
            projections[lang] = []

            for layer in range(n_layers):
                # Get embeddings for this layer and language
                # [n_tokens, d_model]
                layer_embeddings = hidden_space_by_language[lang][layer, :, :]
                # [d_model]
                layer_pca_means = self.pca_means[layer]
                # [n_tokens, d_model]
                centered_layer_embeddings = layer_embeddings - layer_pca_means
                # [n_components, d_model]
                layer_pca_components = self.pca_components[layer][:n_components]
                # [n_tokens, n_components]
                projection = centered_layer_embeddings @ layer_pca_components.T
                projections[lang].append(projection)

            projections[lang] = np.array(projections[lang])

        return projections
