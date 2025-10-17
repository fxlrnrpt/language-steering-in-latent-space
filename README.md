# Lexical subspaces in LLMs

Multilingual Large Language Models (LLMs) often exhibit unintended code-switching, reducing reliability in downstream tasks. We propose latent-space language steering, a lightweight inference-time method that identifies language directions via PCA on parallel translations and steers token embeddings along these axes to control language identity. Our approach mitigates code-switching while preserving semantics with negligible computational overhead and requires only minimal parallel data for calibration. Empirically, we achieve 95-99\% language classification accuracy using a single principal component and reduce next-token distributional divergence by up to 42\% across multiple language pairs on Qwen2.5 and Llama-3.2 models. We further analyze the layer-wise evolution of language representations, revealing that language identity concentrates in final layers with near-perfect linear separability. Code and data are released for reproducibility.

## Data

All datasets can be found in `data` folder.

## How to run

- Install [uv](https://docs.astral.sh/uv/)
- Run `uv sync`
- Start a Jupyter notebook with one of the experiments in `src/experiments`

## Cite

```
@misc{goncharov2025languagesteeringlatentspace,
      title={Language steering in latent space to mitigate unintended code-switching}, 
      author={Andrey Goncharov and Nikolai Kondusov and Alexey Zaytsev},
      year={2025},
      eprint={2510.13849},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.13849}, 
}
```
