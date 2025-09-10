from lexical_reflections.pca.process_hidden_space import extract_pca_components, project_onto_pca
from lexical_reflections.pca.visualize import visualize_explained_variance


def find_lexical_subspace_intersection(hidden_space_by_language):
    pca_components, pca_means, explained_variance_ratios = extract_pca_components(hidden_space_by_language)
    projections = project_onto_pca(hidden_space_by_language, pca_components, pca_means)

    visualize_explained_variance(pca_components, explained_variance_ratios)

    return projections, hidden_space_by_language, (pca_components, pca_means, explained_variance_ratios)
