"""Thread-safe prediction, generation, and trace inspection services."""

from __future__ import annotations

import math
import threading
from typing import Any, Iterable

import torch

from .checkpoint import (
    DEFAULT_CHECKPOINT_FILENAME,
    CheckpointManager,
    LoadedCheckpoint,
)
from .config import ModelConfig
from .model import TinyDecoderLanguageModel
from .tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SPECIAL_TOKENS, WordPunctuationTokenizer


MAX_PROMPT_CHARACTERS = 500
MAX_GENERATED_TOKENS = 50


class ModelNotLoadedError(RuntimeError):
    pass


class InferenceValidationError(ValueError):
    pass


def choose_device(preference: str = "auto") -> torch.device:
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()) else "cpu")


def _validate_temperature(temperature: object) -> float:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise InferenceValidationError("temperature must be a finite number greater than zero")
    return float(temperature)


def _validate_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InferenceValidationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("model produced a non-finite value")
    return round(number, digits)


def _tensor_values(tensor: torch.Tensor, digits: int = 4) -> Any:
    values = tensor.detach().to(device="cpu").tolist()

    def convert(item: Any) -> Any:
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, bool) or isinstance(item, int):
            return item
        return _rounded(float(item), digits)

    return convert(values)


class TinyModelService:
    """Owns one loaded checkpoint model and exposes evaluation-only helpers."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager | None = None,
        *,
        device: str = "auto",
    ) -> None:
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.device = choose_device(device)
        self._lock = threading.RLock()
        self._model: TinyDecoderLanguageModel | None = None
        self._tokenizer: WordPunctuationTokenizer | None = None
        self._loaded_checkpoint: str | None = None
        self._metadata: dict[str, Any] = {}

    def set_model(
        self,
        model: TinyDecoderLanguageModel,
        tokenizer: WordPunctuationTokenizer,
        *,
        checkpoint_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if model.config.vocab_size != tokenizer.vocabulary_size:
            raise ValueError("model vocabulary size does not match the tokenizer")
        with self._lock:
            self._model = model.to(self.device)
            self._model.eval()
            self._tokenizer = tokenizer
            self._loaded_checkpoint = checkpoint_name
            self._metadata = dict(metadata or {})

    def load(self, filename: str = DEFAULT_CHECKPOINT_FILENAME) -> dict[str, Any]:
        loaded: LoadedCheckpoint = self.checkpoint_manager.load(filename, device=self.device)
        self.set_model(
            loaded.model,
            loaded.tokenizer,
            checkpoint_name=filename,
            metadata=loaded.metadata,
        )
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded_checkpoint = self._loaded_checkpoint
            model = self._model
            tokenizer = self._tokenizer
            loaded_metadata = dict(self._metadata)
        checkpoint_status = self.checkpoint_manager.status(
            loaded_checkpoint or DEFAULT_CHECKPOINT_FILENAME
        )
        return {
            **checkpoint_status,
            "loaded": model is not None,
            "loaded_checkpoint": loaded_checkpoint,
            "device": str(self.device),
            "architecture": "pre_norm_decoder",
            "model_config": model.config.to_dict() if model is not None else None,
            "vocabulary_size": tokenizer.vocabulary_size if tokenizer is not None else None,
            "loaded_metadata": loaded_metadata,
        }

    def _snapshot(self) -> tuple[TinyDecoderLanguageModel, WordPunctuationTokenizer]:
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise ModelNotLoadedError("no trained tiny-model checkpoint is loaded")
            return self._model, self._tokenizer

    @staticmethod
    def _validate_text(text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            raise InferenceValidationError("text must not be empty")
        if len(text) > MAX_PROMPT_CHARACTERS:
            raise InferenceValidationError(f"text must contain at most {MAX_PROMPT_CHARACTERS} characters")
        return text.strip()

    @staticmethod
    def _prompt_ids(
        text: str,
        tokenizer: WordPunctuationTokenizer,
        config: ModelConfig,
    ) -> tuple[list[int], bool]:
        token_ids = tokenizer.encode(text, add_bos=True)
        if len(token_ids) <= config.context_length:
            return token_ids, False
        # Retain the explicit BOS marker and the most recent usable context.
        return [tokenizer.bos_token_id, *token_ids[-(config.context_length - 1) :]], True

    @staticmethod
    def _prediction_rows(
        logits: torch.Tensor,
        tokenizer: WordPunctuationTokenizer,
        top_k: int,
        temperature: float,
        *,
        excluded_ids: Iterable[int] = (),
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        adjusted = logits.detach().to(dtype=torch.float32, device="cpu") / temperature
        for token_id in excluded_ids:
            adjusted[int(token_id)] = -torch.inf
        probabilities = torch.softmax(adjusted, dim=-1)
        eligible = int(torch.isfinite(adjusted).sum().item())
        count = min(top_k, eligible)
        top_probabilities, top_ids = torch.topk(probabilities, k=count)
        rows = [
            {
                "token": tokenizer.id_to_token[int(token_id)],
                "token_id": int(token_id),
                "logit": _rounded(float(adjusted[int(token_id)] * temperature), 6),
                "probability": _rounded(float(probability), 6),
            }
            for probability, token_id in zip(top_probabilities.tolist(), top_ids.tolist())
        ]
        return probabilities, rows

    def predict(self, text: str, *, top_k: int = 5, temperature: float = 1.0) -> dict[str, Any]:
        clean_text = self._validate_text(text)
        model, tokenizer = self._snapshot()
        validated_top_k = _validate_integer(top_k, "top_k", 1, tokenizer.vocabulary_size)
        validated_temperature = _validate_temperature(temperature)
        token_ids, truncated = self._prompt_ids(clean_text, tokenizer, model.config)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1]
        probabilities, predictions = self._prediction_rows(
            logits,
            tokenizer,
            validated_top_k,
            validated_temperature,
        )
        return {
            "input_text": clean_text,
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_ids": token_ids,
            "truncated": truncated,
            "top_k": validated_top_k,
            "temperature": validated_temperature,
            "predictions": predictions,
            "probability_sum": _rounded(float(probabilities.sum()), 6),
            "probability_label": "model probability",
        }

    def generate(
        self,
        text: str,
        *,
        max_new_tokens: int = 8,
        temperature: float = 1.0,
        top_k: int = 5,
        strategy: str = "greedy",
        seed: int = 42,
    ) -> dict[str, Any]:
        clean_text = self._validate_text(text)
        model, tokenizer = self._snapshot()
        maximum = _validate_integer(max_new_tokens, "max_new_tokens", 1, MAX_GENERATED_TOKENS)
        validated_top_k = _validate_integer(top_k, "top_k", 1, tokenizer.vocabulary_size)
        validated_temperature = _validate_temperature(temperature)
        if strategy not in {"greedy", "sample"}:
            raise InferenceValidationError("strategy must be 'greedy' or 'sample'")
        if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**31) <= seed <= 2**31 - 1:
            raise InferenceValidationError("seed must be a supported integer")

        original_prompt_ids, initially_truncated = self._prompt_ids(clean_text, tokenizer, model.config)
        rolling_ids = list(original_prompt_ids)
        visible_generated_ids: list[int] = []
        steps: list[dict[str, Any]] = []
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model.eval()
        stop_reason = "max_new_tokens"
        excluded_ids = (tokenizer.pad_token_id, tokenizer.bos_token_id)

        for step_index in range(maximum):
            context_ids = rolling_ids
            if len(context_ids) > model.config.context_length:
                context_ids = [tokenizer.bos_token_id, *context_ids[-(model.config.context_length - 1) :]]
            input_tensor = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = model(input_tensor).logits[0, -1]
            probabilities, top_predictions = self._prediction_rows(
                logits,
                tokenizer,
                validated_top_k,
                validated_temperature,
                excluded_ids=excluded_ids,
            )
            if strategy == "greedy":
                chosen_id = int(torch.argmax(probabilities).item())
            else:
                chosen_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            chosen_token = tokenizer.id_to_token[chosen_id]
            chosen_probability = _rounded(float(probabilities[chosen_id]), 6)
            is_eos = chosen_id == tokenizer.eos_token_id
            steps.append(
                {
                    "step": step_index + 1,
                    "chosen_token": chosen_token,
                    "chosen_token_id": chosen_id,
                    "chosen_probability": chosen_probability,
                    "top_predictions": top_predictions,
                    "is_eos": is_eos,
                }
            )
            rolling_ids.append(chosen_id)
            if is_eos:
                stop_reason = "eos"
                break
            # PAD/BOS have zero probability, and EOS is retained only as a
            # stopping event, so visible output contains no forbidden markers.
            if chosen_token not in SPECIAL_TOKENS:
                visible_generated_ids.append(chosen_id)

        prompt_without_bos = original_prompt_ids[1:]
        generated_text = tokenizer.decode([*prompt_without_bos, *visible_generated_ids])
        return {
            "input_text": clean_text,
            "input_tokens": tokenizer.convert_ids_to_tokens(original_prompt_ids),
            "input_token_ids": original_prompt_ids,
            "truncated": initially_truncated,
            "generated_text": generated_text,
            "generated_tokens": tokenizer.convert_ids_to_tokens(visible_generated_ids),
            "generated_token_ids": visible_generated_ids,
            "strategy": strategy,
            "seed": seed,
            "temperature": validated_temperature,
            "top_k": validated_top_k,
            "max_new_tokens": maximum,
            "stop_reason": stop_reason,
            "steps": steps,
            "probability_label": "model probability",
        }

    def inspect(
        self,
        text: str,
        *,
        layer: int = 0,
        head: int = 0,
        query_token: int | None = None,
        key_token: int | None = None,
        hidden_dimension: int = 0,
        top_k: int = 5,
    ) -> dict[str, Any]:
        clean_text = self._validate_text(text)
        model, tokenizer = self._snapshot()
        layer_index = _validate_integer(layer, "layer", 0, model.config.number_of_layers - 1)
        head_index = _validate_integer(head, "head", 0, model.config.number_of_heads - 1)
        dimension_index = _validate_integer(hidden_dimension, "hidden_dimension", 0, model.config.d_model - 1)
        validated_top_k = _validate_integer(top_k, "top_k", 1, tokenizer.vocabulary_size)
        token_ids, truncated = self._prompt_ids(clean_text, tokenizer, model.config)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        sequence_length = len(token_ids)
        query_index = sequence_length - 1 if query_token is None else _validate_integer(
            query_token, "query_token", 0, sequence_length - 1
        )
        key_index = query_index if key_token is None else _validate_integer(
            key_token, "key_token", 0, sequence_length - 1
        )

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor, return_trace=True)
        assert output.trace is not None
        trace = output.trace
        selected = trace.layers[layer_index]
        logits = trace.vocabulary_logits[0, -1]
        probabilities, top_predictions = self._prediction_rows(logits, tokenizer, validated_top_k, 1.0)
        head_dimension = model.config.head_dimension
        query_vector = selected.query[0, head_index, query_index]
        key_vector = selected.key[0, head_index, key_index]
        products = query_vector * key_vector
        raw_score = float(products.sum())
        masked = key_index > query_index
        connection_rows = [
            {
                "key_token": tokens[index],
                "key_position": index,
                "attention_weight": _rounded(
                    float(selected.attention_probabilities[0, head_index, query_index, index]), 6
                ),
                "causally_available": index <= query_index,
            }
            for index in range(sequence_length)
        ]

        return {
            "input_text": clean_text,
            "tokens": tokens,
            "token_ids": token_ids,
            "truncated": truncated,
            "architecture": "modern pre-normalization decoder block",
            "attention_note": "Attention patterns show where a head reads information from; they are not a complete explanation of a prediction.",
            "model_config": model.config.to_dict(),
            "selection": {
                "layer": layer_index,
                "head": head_index,
                "query_token": query_index,
                "key_token": key_index,
                "hidden_dimension": dimension_index,
            },
            "shapes": {
                "token_embeddings": list(trace.token_embeddings.shape),
                "position_embeddings": list(trace.position_embeddings.shape),
                "combined_embeddings": list(trace.combined_embeddings.shape),
                "query": list(selected.query.shape),
                "key": list(selected.key.shape),
                "value": list(selected.value.shape),
                "attention_scores": list(selected.raw_attention_scores.shape),
                "attention_probabilities": list(selected.attention_probabilities.shape),
                "head_context_vectors": list(selected.head_context_vectors.shape),
                "feed_forward_activations": list(selected.gelu_activations.shape),
                "final_hidden_states": list(trace.final_hidden_states.shape),
                "vocabulary_logits": list(trace.vocabulary_logits.shape),
            },
            "token_embeddings": _tensor_values(trace.token_embeddings[0]),
            "position_embeddings": _tensor_values(trace.position_embeddings[0]),
            "combined_embeddings": _tensor_values(trace.combined_embeddings[0]),
            "layer_trace": {
                "normalized_attention_input": _tensor_values(selected.normalized_attention_input[0]),
                "query": _tensor_values(selected.query[0, head_index]),
                "key": _tensor_values(selected.key[0, head_index]),
                "value": _tensor_values(selected.value[0, head_index]),
                "raw_attention_scores": _tensor_values(selected.raw_attention_scores[0, head_index]),
                "scaled_attention_scores": _tensor_values(selected.scaled_attention_scores[0, head_index]),
                "causal_mask": _tensor_values(selected.causal_mask),
                "attention_probabilities": _tensor_values(selected.attention_probabilities[0, head_index]),
                "head_context_vectors": _tensor_values(selected.head_context_vectors[0, head_index]),
                "concatenated_attention_output": _tensor_values(selected.concatenated_attention_output[0]),
                "projected_attention_output": _tensor_values(selected.projected_attention_output[0]),
                "attention_residual_output": _tensor_values(selected.attention_residual_output[0]),
                "normalized_feed_forward_input": _tensor_values(selected.normalized_feed_forward_input[0]),
                "feed_forward_pre_activations": _tensor_values(selected.feed_forward_pre_activations[0]),
                "gelu_activations": _tensor_values(selected.gelu_activations[0]),
                "feed_forward_output": _tensor_values(selected.feed_forward_output[0]),
                "block_output": _tensor_values(selected.block_output[0]),
            },
            "selected_attention_calculation": {
                "query_token": tokens[query_index],
                "query_position": query_index,
                "key_token": tokens[key_index],
                "key_position": key_index,
                "query_vector": _tensor_values(query_vector),
                "key_vector": _tensor_values(key_vector),
                "products": _tensor_values(products),
                "raw_score": _rounded(raw_score, 6),
                "scale_factor": _rounded(math.sqrt(head_dimension), 6),
                "scaled_score": _rounded(raw_score / math.sqrt(head_dimension), 6),
                "causally_masked": masked,
                "attention_probability": _rounded(
                    float(selected.attention_probabilities[0, head_index, query_index, key_index]), 6
                ),
            },
            "token_connections": connection_rows,
            "selected_hidden_values": {
                "dimension": dimension_index,
                "final_hidden_states": [
                    _rounded(float(value), 4)
                    for value in trace.final_hidden_states[0, :, dimension_index]
                ],
            },
            "final_hidden_states": _tensor_values(trace.final_hidden_states[0]),
            "vocabulary_logits": _tensor_values(trace.vocabulary_logits[0]),
            "vocabulary_probabilities": _tensor_values(trace.vocabulary_probabilities[0], 6),
            "top_predictions": top_predictions,
            "probability_sum": _rounded(float(probabilities.sum()), 6),
        }


# Service alias for concise imports in FastAPI wiring.
InferenceService = TinyModelService
