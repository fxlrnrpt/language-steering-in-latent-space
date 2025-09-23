from typing import Optional

import torch
from sklearn.decomposition import PCA


class PCASteering:
    def __init__(self, n_components=10) -> None:
        self.n_components: int = n_components
        # [n_layers, n_components, d_model]
        self.pca_components: Optional[torch.Tensor] = None
        # [n_layers, d_model]
        self.pca_means: Optional[torch.Tensor] = None
        # [n_layers, n_components]
        self.explained_variance_ratios: Optional[torch.Tensor] = None
        # { [lang]: [n_layers, n_components] }
        self.lang_vectors_by_component: dict[str, torch.Tensor] = {}

    def fit(self, hidden_space_by_language):
        """
        :param hidden_space_by_language: { [lang]: torch.Tensor([n_layers, n_tokens, d_model]) }
        """
        # n_layers, n_tokens (across all langs), d_model
        combined_embeddings = torch.concat(list(hidden_space_by_language.values()), dim=1)
        n_layers, _, d_model = combined_embeddings.shape

        pca_components = []
        pca_means = []
        explained_variance_ratios = []

        for layer in range(n_layers):
            pca = PCA(n_components=self.n_components)
            pca.fit_transform(combined_embeddings[layer].numpy(force=True))

            pca_components.append(pca.components_)  # Principal components [n_components, d_model]
            pca_means.append(pca.mean_)
            explained_variance_ratios.append(pca.explained_variance_ratio_)

        self.pca_components = torch.tensor(pca_components)
        self.pca_means = torch.tensor(pca_means)
        self.explained_variance_ratios = torch.tensor(explained_variance_ratios)

        projections_by_language = self.project(hidden_space_by_language)
        for lang, projections in projections_by_language.items():
            lang_vectors_by_component = []
            for layer in range(n_layers):
                projections_layer = projections[layer]
                projections_layer_mean = projections_layer.mean(dim=0).squeeze()
                assert projections_layer_mean.shape == (self.n_components,)
                lang_vectors_by_component.append(projections_layer_mean)
            self.lang_vectors_by_component[lang] = torch.stack(lang_vectors_by_component)

        return self

    def project(self, hidden_space_by_language):
        """
        Project each language's embeddings onto the common subspace.

        Parameters:

        - hidden_space_by_language: { [lang]: torch.Tensor([n_layers, n_tokens, d_model]) }

        Returns:
        { [lang]: torch.Tensor([n_layers, n_tokens, n_components]) }
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
                layer_pca_components = self.pca_components[layer]
                # [n_tokens, n_components]
                projection = centered_layer_embeddings @ layer_pca_components.T
                projections[lang].append(projection)

            projections[lang] = torch.stack(projections[lang], dim=0)

        return projections

    def steer(self, X: torch.Tensor, direction: tuple[str, str], layer: int):
        """
        :param X: [batch_size, seq_length, d_model]
        :param direction: pair of the languages from hidden_space_by_language that you passed to `fit` (language from and language to)
        """
        assert self.pca_components is not None
        assert self.pca_means is not None

        B, N, C = X.shape
        lang_from, lang_to = direction

        centered_X = X - self.pca_means[layer]
        # [n_components, d_model]
        layer_pca_components = self.pca_components[layer]
        # [B, N, n_components]
        projected_X = centered_X @ layer_pca_components.T
        assert projected_X.shape == (B, N, self.n_components)

        # [n_components]
        source_direction = self.lang_vectors_by_component[lang_from][layer]
        target_direction = self.lang_vectors_by_component[lang_to][layer]
        assert source_direction.shape == (self.n_components,)
        assert target_direction.shape == (self.n_components,)

        eps = 1e-8
        scale = target_direction / (source_direction + eps)
        steered_projected_X = projected_X * scale

        # [B, N, C]
        steered_X = steered_projected_X @ layer_pca_components + self.pca_means[layer]
        return steered_X
