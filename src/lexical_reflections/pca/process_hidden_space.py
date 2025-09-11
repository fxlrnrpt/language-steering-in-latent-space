import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def extract_pca_components(hidden_space_by_language, n_components=10):
    """
    Extract common subspace across languages using PCA.
    This identifies directions that explain most variance in both languages.

    hidden_space_by_language: { [lang]: np.array([n_layers, n_entries, d_model]) }
    """
    combined_embeddings = np.concat(list(hidden_space_by_language.values()), axis=1)

    n_layers = combined_embeddings.shape[0]

    pca_components = []
    pca_means = []
    explained_variance_ratios = []

    for layer in range(n_layers):
        # Apply PCA to find common directions
        pca = PCA(n_components=n_components)
        pca.fit_transform(combined_embeddings[layer])

        pca_components.append(pca.components_)  # Principal components [n_components, d_model]
        pca_means.append(pca.mean_)
        explained_variance_ratios.append(pca.explained_variance_ratio_)

    return pca_components, pca_means, explained_variance_ratios


def project_onto_pca(hidden_space_by_language, pca_components, pca_means):
    """
    Project each language's embeddings onto the common subspace.

    hidden_space_by_language: { [lang]: np.array([n_layers, n_tokens, d_model]) }
    """
    n_layers = len(pca_components)
    projections = {}
    cosine_similarities = {k: [] for k in hidden_space_by_language.keys()}

    for lang in hidden_space_by_language:
        projections[lang] = []
        cosine_similarities[lang] = []

        for layer in range(n_layers):
            # Get embeddings for this layer and language
            layer_embeddings = hidden_space_by_language[lang][layer, :, :]  # [n_tokens, d_model]
            centered_layer_embeddings = layer_embeddings - pca_means[layer]
            projection = pca_components[layer] @ centered_layer_embeddings.T
            projections[lang].append(projection)

            projection_in_original_space = projection[:1].T @ pca_components[layer][:1] + pca_means[layer][:1]
            projection_cosine_similarity = cosine_similarity(layer_embeddings, projection_in_original_space)
            cosine_similarities[lang].append(projection_cosine_similarity)

    return projections, cosine_similarities
