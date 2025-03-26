# Lexical subspaces in LLMs

What language do you think in? Is it even a language or some higher level of abstraction? What about LLMs?

Ever since I learned English (to complement my native Russian) I have always been puzzled when people tried to reason which language we think in. I had a clear feeling that, personally, I do not think in a certain language, — rather in some abstract concepts that later find their meaning in words when I try to convey them to another person. At times, it is easier to express some ideas in a certain language than the other, so I quite frequently jump between them.

In mathematical terms, it seems that there is a high-dimensional space of abstract ideas that is being projected to the lower dimensional lexical spaces. The fact that sometimes it is easier to express some ideas in one language than the other is evidence of the lower dimensionality of the lexical spaces. When we say that “it is easier” we usually mean that the projection captures more information and, therefore, the reconstruction of the original idea from the projection has a smaller loss.

This projects attempts to explore the idea of identifying and separating lexical subspaces.

## Experiments

1. PCA subspace identification ([Qwen](src/experiments/pca_classifier_qwen.ipynb), [mGPT](src/experiments/pca_classifier_qwen.ipynb))
2. Language classifier based on hidden spaces ([Qwen](src/experiments/pca_classifier_qwen.ipynb))

## How to run

- Install [uv](https://docs.astral.sh/uv/)
- Run `uv sync`
- Start a Jupyter notebook with one of the experiments