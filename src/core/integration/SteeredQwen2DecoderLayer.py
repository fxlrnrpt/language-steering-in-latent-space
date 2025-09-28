from typing import Optional

import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from core.steering.pca import PCASteering


class SteeredQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int, steering_module: PCASteering):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.steering_module = steering_module
        self.steering_direction = 0.0
        self.source_projected_hidden_states: Optional[torch.Tensor] = None

    def forward(self, *args, **kwargs):
        hidden_states = super().forward(*args, **kwargs)
        if self.steering_direction != 0.0:
            steered_hidden_states, projected_hidden_states = self.steering_module.steer(
                hidden_states, self.layer_idx, self.source_projected_hidden_states, self.steering_direction
            )
            if self.source_projected_hidden_states is None:
                B, N, C = projected_hidden_states.shape
                self.source_projected_hidden_states = projected_hidden_states.view(B * N, C).mean(dim=0, keepdim=True)
            return steered_hidden_states
        return hidden_states

    def set_steering_direction(self, steering_direction: float):
        self.steering_direction = steering_direction

    def reset(self):
        self.source_projected_hidden_states = None

    def autosteer(self, source_lang: str, target_lang: str):
        source_lang_vectors = self.steering_module.lang_vectors_by_component[source_lang][self.layer_idx][0]
        target_lang_vectors = self.steering_module.lang_vectors_by_component[target_lang][self.layer_idx][0]
        direction = target_lang_vectors / source_lang_vectors - 1
        self.steering_direction = direction.cpu().detach().tolist()
