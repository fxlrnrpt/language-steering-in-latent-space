import numpy as np
from sklearn.decomposition import PCA


def extract_pca_components(hidden_space_by_language, n_components=10):
    """
    Extract common subspace across languages using PCA.
    This identifies directions that explain most variance in both languages.
    """
    languages = list(hidden_space_by_language.keys())

    n_layers = hidden_space_by_language[languages[0]].shape[2]
    d_model = hidden_space_by_language[languages[0]].shape[0]
    n_prompts = hidden_space_by_language[languages[0]].shape[1]

    pca_components = []
    explained_variance_ratios = []

    for layer in range(n_layers):
        # Concatenate embeddings from both languages
        combined_embeddings = np.zeros((d_model, n_prompts * len(languages)))

        for i, lang in enumerate(languages):
            combined_embeddings[:, i * n_prompts : (i + 1) * n_prompts] = hidden_space_by_language[lang][:, :, layer]

        # Apply PCA to find common directions
        pca = PCA(n_components=n_components)
        # Transpose to get [n_samples, n_features] format expected by sklearn
        pca.fit_transform(combined_embeddings.T)

        pca_components.append(pca.components_)  # Principal components [n_components, d_model]
        explained_variance_ratios.append(pca.explained_variance_ratio_)

    return pca_components, explained_variance_ratios


def project_onto_pca(hidden_space_by_language, pca_components):
    """
    Project each language's embeddings onto the common subspace.
    """
    n_layers = len(pca_components)
    projections = {}

    for lang in hidden_space_by_language:
        projections[lang] = []

        for layer in range(n_layers):
            # Get embeddings for this layer and language
            layer_embeddings = hidden_space_by_language[lang][:, :, layer]  # [d_model, n_prompts]
            projection = pca_components[layer] @ layer_embeddings
            projections[lang].append(projection)

    return projections
