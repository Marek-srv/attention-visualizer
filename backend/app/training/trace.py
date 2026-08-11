"""Detached trace structures produced only during explicit model inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def detached_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Return a trace-safe tensor with no graph and no device dependency."""

    return tensor.detach().to(device="cpu").clone()


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


@dataclass(slots=True)
class LayerTrace:
    normalized_attention_input: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    raw_attention_scores: torch.Tensor
    scaled_attention_scores: torch.Tensor
    causal_mask: torch.Tensor
    attention_probabilities: torch.Tensor
    head_context_vectors: torch.Tensor
    concatenated_attention_output: torch.Tensor
    projected_attention_output: torch.Tensor
    attention_residual_output: torch.Tensor
    normalized_feed_forward_input: torch.Tensor
    feed_forward_pre_activations: torch.Tensor
    gelu_activations: torch.Tensor
    feed_forward_output: torch.Tensor
    block_output: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def __getitem__(self, name: str) -> torch.Tensor:
        return getattr(self, name)


@dataclass(slots=True)
class ModelTrace:
    token_ids: torch.Tensor
    token_embeddings: torch.Tensor
    position_embeddings: torch.Tensor
    combined_embeddings: torch.Tensor
    layers: list[LayerTrace]
    final_hidden_states: torch.Tensor
    vocabulary_logits: torch.Tensor
    vocabulary_probabilities: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "token_embeddings": self.token_embeddings,
            "position_embeddings": self.position_embeddings,
            "combined_embeddings": self.combined_embeddings,
            "layers": [layer.as_dict() for layer in self.layers],
            "final_hidden_states": self.final_hidden_states,
            "vocabulary_logits": self.vocabulary_logits,
            "vocabulary_probabilities": self.vocabulary_probabilities,
        }

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)
