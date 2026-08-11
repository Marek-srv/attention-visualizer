"""A compact, explicit PyTorch decoder-only Transformer language model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .trace import LayerTrace, ModelTrace, detached_cpu


@dataclass(slots=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    trace: ModelTrace | None = None

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)


@dataclass(slots=True)
class AttentionIntermediates:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    raw_scores: torch.Tensor
    scaled_scores: torch.Tensor
    causal_mask: torch.Tensor
    probabilities: torch.Tensor
    head_contexts: torch.Tensor
    concatenated_output: torch.Tensor
    projected_output: torch.Tensor


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head attention with a visible lower-triangular causal mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.number_of_heads = config.number_of_heads
        self.head_dimension = config.head_dimension
        self.query_projection = nn.Linear(config.d_model, config.d_model, bias=True)
        self.key_projection = nn.Linear(config.d_model, config.d_model, bias=True)
        self.value_projection = nn.Linear(config.d_model, config.d_model, bias=True)
        self.output_projection = nn.Linear(config.d_model, config.d_model, bias=True)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        causal_mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, AttentionIntermediates | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        value = self._split_heads(self.value_projection(hidden_states))

        raw_scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_scores = raw_scores / math.sqrt(self.head_dimension)
        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        allowed = causal_mask.view(1, 1, sequence_length, sequence_length)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask must have shape batch_size x sequence_length")
            # Padding is excluded as a key. Each real example starts with BOS,
            # so every row always retains at least one finite attention score.
            allowed = allowed & attention_mask.to(dtype=torch.bool).view(batch_size, 1, 1, sequence_length)
        masked_scores = scaled_scores.masked_fill(~allowed, torch.finfo(scaled_scores.dtype).min)
        probabilities = torch.softmax(masked_scores, dim=-1)
        # Set prohibited locations to exact zero (rather than merely tiny) and
        # re-normalize for stable, directly testable causal rows.
        probabilities = probabilities.masked_fill(~allowed, 0.0)
        probability_sums = probabilities.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        probabilities = probabilities / probability_sums
        dropped_probabilities = self.attention_dropout(probabilities)

        head_contexts = torch.matmul(dropped_probabilities, value)
        concatenated = head_contexts.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        projected = self.output_dropout(self.output_projection(concatenated))

        intermediates = None
        if return_intermediates:
            intermediates = AttentionIntermediates(
                query=query,
                key=key,
                value=value,
                raw_scores=raw_scores,
                scaled_scores=scaled_scores,
                causal_mask=causal_mask,
                probabilities=probabilities,
                head_contexts=head_contexts,
                concatenated_output=concatenated,
                projected_output=projected,
            )
        return projected, intermediates


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.d_model, config.feed_forward_dimension)
        self.output_projection = nn.Linear(config.feed_forward_dimension, config.d_model)
        self.activation = nn.GELU()
        self.hidden_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        pre_activations = self.input_projection(hidden_states)
        activations = self.activation(pre_activations)
        output = self.output_dropout(self.output_projection(self.hidden_dropout(activations)))
        intermediates = (pre_activations, activations) if return_intermediates else None
        return output, intermediates


class PreNormDecoderBlock(nn.Module):
    """Modern pre-norm block: residuals wrap normalized attention and FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadCausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForwardNetwork(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, LayerTrace | None]:
        normalized_attention_input = self.layer_norm_1(hidden_states)
        attention_output, attention_values = self.attention(
            normalized_attention_input,
            attention_mask,
            return_intermediates=return_trace,
        )
        attention_residual = hidden_states + attention_output
        normalized_feed_forward_input = self.layer_norm_2(attention_residual)
        feed_forward_output, feed_forward_values = self.feed_forward(
            normalized_feed_forward_input,
            return_intermediates=return_trace,
        )
        block_output = attention_residual + feed_forward_output

        layer_trace = None
        if return_trace:
            assert attention_values is not None and feed_forward_values is not None
            pre_activations, gelu_activations = feed_forward_values
            layer_trace = LayerTrace(
                normalized_attention_input=detached_cpu(normalized_attention_input),
                query=detached_cpu(attention_values.query),
                key=detached_cpu(attention_values.key),
                value=detached_cpu(attention_values.value),
                raw_attention_scores=detached_cpu(attention_values.raw_scores),
                scaled_attention_scores=detached_cpu(attention_values.scaled_scores),
                causal_mask=detached_cpu(attention_values.causal_mask),
                attention_probabilities=detached_cpu(attention_values.probabilities),
                head_context_vectors=detached_cpu(attention_values.head_contexts),
                concatenated_attention_output=detached_cpu(attention_values.concatenated_output),
                projected_attention_output=detached_cpu(attention_values.projected_output),
                attention_residual_output=detached_cpu(attention_residual),
                normalized_feed_forward_input=detached_cpu(normalized_feed_forward_input),
                feed_forward_pre_activations=detached_cpu(pre_activations),
                gelu_activations=detached_cpu(gelu_activations),
                feed_forward_output=detached_cpu(feed_forward_output),
                block_output=detached_cpu(block_output),
            )
        return block_output, layer_trace


class TinyDecoderLanguageModel(nn.Module):
    """Two-block-by-default causal language model for local education."""

    architecture = "pre_norm_decoder"

    def __init__(self, config: ModelConfig, *, pad_token_id: int = 0) -> None:
        super().__init__()
        if not 0 <= pad_token_id < config.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")
        self.config = config
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([PreNormDecoderBlock(config) for _ in range(config.number_of_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        with torch.no_grad():
            self.token_embedding.weight[pad_token_id].zero_()

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        labels: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> CausalLMOutput:
        if labels is not None:
            if targets is not None:
                raise ValueError("provide either targets or labels, not both")
            targets = labels
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape batch_size x sequence_length")
        batch_size, sequence_length = input_ids.shape
        if sequence_length < 1:
            raise ValueError("input_ids must contain at least one token")
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.config.context_length}"
            )
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, sequence_length)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        combined_embeddings = token_embeddings + position_embeddings
        hidden_states = self.embedding_dropout(combined_embeddings)

        layer_traces: list[LayerTrace] = []
        for block in self.blocks:
            hidden_states, layer_trace = block(
                hidden_states,
                attention_mask,
                return_trace=return_trace,
            )
            if layer_trace is not None:
                layer_traces.append(layer_trace)

        final_hidden_states = self.final_layer_norm(hidden_states)
        logits = self.language_model_head(final_hidden_states)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.to(dtype=torch.long).reshape(-1),
                ignore_index=self.pad_token_id,
            )

        model_trace = None
        if return_trace:
            model_trace = ModelTrace(
                token_ids=detached_cpu(input_ids),
                token_embeddings=detached_cpu(token_embeddings),
                position_embeddings=detached_cpu(position_embeddings),
                combined_embeddings=detached_cpu(combined_embeddings),
                layers=layer_traces,
                final_hidden_states=detached_cpu(final_hidden_states),
                vocabulary_logits=detached_cpu(logits),
                vocabulary_probabilities=detached_cpu(torch.softmax(logits, dim=-1)),
            )
        return CausalLMOutput(logits=logits, loss=loss, trace=model_trace)

    @property
    def transformer_blocks(self) -> nn.ModuleList:
        return self.blocks

    @property
    def lm_head(self) -> nn.Linear:
        return self.language_model_head


# Common names retained as aliases to keep route code and focused tests concise.
TinyTransformerLanguageModel = TinyDecoderLanguageModel
TinyTransformerLM = TinyDecoderLanguageModel
DecoderBlock = PreNormDecoderBlock
