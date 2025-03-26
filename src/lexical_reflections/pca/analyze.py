from lexical_reflections.gather_data.hidden_space import collect_hidden_space_by_language
from lexical_reflections.pca.process_hidden_space import extract_pca_components, project_onto_pca
from lexical_reflections.pca.visualize import visualize_explained_variance, visualize_projections


def find_lexical_subspace_intersection(model, activation_cache):
    # { [lang]: np.array([d_model, n_prompts, n_layers]) }
    hidden_space_by_language = collect_hidden_space_by_language(model, activation_cache)

    pca_components, pca_means, explained_variance_ratios = extract_pca_components(hidden_space_by_language)
    projections = project_onto_pca(hidden_space_by_language, pca_components, pca_means)

    visualize_explained_variance(pca_components, explained_variance_ratios)
    visualize_projections(hidden_space_by_language, projections)
