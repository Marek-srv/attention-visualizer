"""Lazy, optional Hugging Face causal-language-model inspection service.

This module deliberately imports neither PyTorch nor Transformers at import time.
Importing ``app.main`` therefore never downloads a model and continues to work
when the optional pretrained-model dependencies are not installed.  A model is
loaded only by :meth:`PretrainedModelService.load`, ``inspect``, or ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
import threading
from typing import Any


DEFAULT_PRETRAINED_MODEL = "sshleifer/tiny-gpt2"
MODEL_NAME_ENV = "PRETRAINED_MODEL_NAME"
DEVICE_ENV = "PRETRAINED_DEVICE"
MAX_PROMPT_CHARACTERS = 500
DEFAULT_TOP_K = 5
ROUND_DIGITS = 6


class PretrainedServiceError(RuntimeError):
    """Base exception safe for a FastAPI route to return to a client."""

    status_code = 503
    error_code = "pretrained_model_error"

    def as_detail(self) -> dict[str, str]:
        return {"code": self.error_code, "message": str(self)}


class PretrainedValidationError(PretrainedServiceError, ValueError):
    """The caller supplied an invalid prompt or selection."""

    status_code = 422
    error_code = "pretrained_validation_error"


class PretrainedDependencyError(PretrainedServiceError):
    """The optional PyTorch/Transformers packages are unavailable."""

    error_code = "pretrained_dependencies_unavailable"


class PretrainedLoadError(PretrainedServiceError):
    """The configured model could not be obtained or initialized."""

    error_code = "pretrained_model_load_failed"


class PretrainedCompatibilityError(PretrainedServiceError):
    """The model/runtime cannot expose the attention data this UI requires."""

    error_code = "pretrained_attention_unavailable"


@dataclass(slots=True)
class _ModelBundle:
    """Runtime objects kept out of public responses and module-level imports."""

    torch: Any
    tokenizer: Any
    model: Any
    model_name: str
    device: Any
    metadata: dict[str, Any]


def _module_available(module_name: str) -> bool:
    """Check an optional dependency without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _rounded_finite(value: Any, *, label: str = "model output") -> float:
    """Convert a scalar tensor/number to a finite, JSON-safe rounded float."""

    if hasattr(value, "item"):
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PretrainedCompatibilityError(f"{label} was not numeric.") from exc
    if not math.isfinite(numeric):
        raise PretrainedCompatibilityError(f"{label} contained NaN or Infinity.")
    rounded = round(numeric, ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "zero or greater" if allow_zero else "greater than zero"
        raise PretrainedValidationError(f"{name} must be an integer {qualifier}.")
    return value


def _validate_prompt(text: Any) -> str:
    if not isinstance(text, str):
        raise PretrainedValidationError("text must be a string.")
    if not text.strip():
        raise PretrainedValidationError("text must not be empty.")
    if len(text) > MAX_PROMPT_CHARACTERS:
        raise PretrainedValidationError(
            f"text must contain at most {MAX_PROMPT_CHARACTERS} characters."
        )
    return text


def _validate_temperature(temperature: Any) -> float:
    if isinstance(temperature, bool):
        raise PretrainedValidationError("temperature must be finite and greater than zero.")
    try:
        numeric = float(temperature)
    except (TypeError, ValueError) as exc:
        raise PretrainedValidationError(
            "temperature must be finite and greater than zero."
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise PretrainedValidationError("temperature must be finite and greater than zero.")
    return numeric


def _config_integer(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _context_limit(tokenizer: Any, config: Any) -> int:
    """Find a real model context limit while ignoring tokenizer sentinel values."""

    candidates: list[int] = []
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)
    # GPT-2-family configs expose 1024.  The conservative fallback is only used
    # for compatible custom configs that omit all conventional field names.
    return min(candidates) if candidates else 1024


def _token_string(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if isinstance(token, list):
        token = token[0] if token else ""
    return str(token)


def _decoded_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False))
    except TypeError:
        # Older Transformers versions do not expose the cleanup keyword here.
        return str(tokenizer.decode([int(token_id)]))


class PretrainedModelService:
    """Thread-safe lazy cache and focused inspector for a small causal model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        configured_name = model_name or os.getenv(MODEL_NAME_ENV, DEFAULT_PRETRAINED_MODEL)
        configured_device = device or os.getenv(DEVICE_ENV, "cpu")
        self.model_name = configured_name.strip() or DEFAULT_PRETRAINED_MODEL
        self.device_name = configured_device.strip().lower() or "cpu"
        self._condition = threading.Condition(threading.RLock())
        self._inference_lock = threading.Lock()
        self._bundle: _ModelBundle | None = None
        self._loading = False
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        """Return cache status without importing dependencies or loading a model."""

        with self._condition:
            bundle = self._bundle
            state = "loaded" if bundle is not None else "loading" if self._loading else (
                "failed" if self._last_error else "not_loaded"
            )
            return {
                "status": state,
                "loaded": bundle is not None,
                "loading": self._loading,
                "model_name": self.model_name,
                "device": bundle.metadata["device"] if bundle else self.device_name,
                "dependencies_available": {
                    "torch": _module_available("torch"),
                    "transformers": _module_available("transformers"),
                },
                "model": dict(bundle.metadata) if bundle else None,
                "error": self._last_error,
            }

    def load(self) -> dict[str, Any]:
        """Load and cache the configured model once, waiting for a concurrent load."""

        waited_for_another_loader = False
        with self._condition:
            if self._bundle is not None:
                return self.status()
            while self._loading:
                waited_for_another_loader = True
                self._condition.wait()
            if self._bundle is not None:
                return self.status()
            if waited_for_another_loader and self._last_error:
                raise PretrainedLoadError(self._last_error)
            self._loading = True
            self._last_error = None

        try:
            bundle = self._load_bundle()
        except PretrainedServiceError as exc:
            with self._condition:
                self._loading = False
                self._last_error = str(exc)
                self._condition.notify_all()
            raise
        except Exception as exc:  # Do not expose a dependency stack trace to API clients.
            error = PretrainedLoadError(
                f"Could not load pretrained model '{self.model_name}'. Check internet access, "
                "the configured model name, and the local Hugging Face cache."
            )
            with self._condition:
                self._loading = False
                self._last_error = str(error)
                self._condition.notify_all()
            raise error from exc

        with self._condition:
            self._bundle = bundle
            self._loading = False
            self._last_error = None
            self._condition.notify_all()
        return self.status()

    def inspect(
        self,
        text: str,
        *,
        layer: int = 0,
        head: int = 0,
        query_index: int | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Return one selected head matrix, query row, and next-token results."""

        prompt = _validate_prompt(text)
        selected_layer = _positive_integer(layer, "layer", allow_zero=True)
        selected_head = _positive_integer(head, "head", allow_zero=True)
        requested_top_k = _positive_integer(top_k, "top_k")
        if query_index is not None:
            selected_query = _positive_integer(query_index, "query_index", allow_zero=True)
        else:
            selected_query = None

        bundle = self._get_or_load_bundle()
        encoded = self._encode_prompt(bundle, prompt)

        with self._inference_lock, bundle.torch.no_grad():
            outputs = bundle.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = getattr(outputs, "attentions", None)
        if attentions is None or len(attentions) == 0:
            raise PretrainedCompatibilityError(
                "The configured model did not return attention tensors. Use a Transformers "
                "version/model that supports output_attentions=True with eager attention."
            )
        if selected_layer >= len(attentions):
            raise PretrainedValidationError(
                f"layer must be between 0 and {len(attentions) - 1} for this model."
            )

        layer_attention = attentions[selected_layer]
        if getattr(layer_attention, "ndim", None) != 4 or layer_attention.shape[0] != 1:
            raise PretrainedCompatibilityError(
                "The model returned an unsupported attention tensor shape; expected "
                "batch x heads x query tokens x key tokens."
            )
        number_of_heads = int(layer_attention.shape[1])
        if selected_head >= number_of_heads:
            raise PretrainedValidationError(
                f"head must be between 0 and {number_of_heads - 1} for layer {selected_layer}."
            )

        token_ids = encoded["token_ids"]
        tokens = encoded["tokens"]
        token_count = len(token_ids)
        actual_query = token_count - 1 if selected_query is None else selected_query
        if actual_query >= token_count:
            raise PretrainedValidationError(
                f"query_index must be between 0 and {token_count - 1} for this prompt."
            )

        selected_matrix_tensor = layer_attention[0, selected_head].detach().to("cpu")
        matrix = [
            [_rounded_finite(value, label="attention weight") for value in row]
            for row in selected_matrix_tensor.tolist()
        ]
        query_row = matrix[actual_query]
        connections = [
            {
                "key_index": key_index,
                "key_token": tokens[key_index],
                "key_token_id": token_ids[key_index],
                "attention_weight": weight,
                "is_future": key_index > actual_query,
            }
            for key_index, weight in enumerate(query_row)
        ]

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise PretrainedCompatibilityError("The configured causal model did not return logits.")
        prediction = self._prediction_values(
            bundle,
            logits[0, -1],
            top_k=requested_top_k,
            temperature=1.0,
        )

        return {
            "model_name": bundle.model_name,
            "device": bundle.metadata["device"],
            "input_text": prompt,
            "tokens": tokens,
            "token_ids": token_ids,
            "token_count": token_count,
            "original_token_count": encoded["original_token_count"],
            "context_truncated": encoded["context_truncated"],
            "selected_layer": selected_layer,
            "selected_head": selected_head,
            "selected_query_index": actual_query,
            "selected_query_token": tokens[actual_query],
            "selected_query_token_id": token_ids[actual_query],
            "attention_shape": [int(value) for value in selected_matrix_tensor.shape],
            "attention_matrix": matrix,
            "attention_row_sum": _rounded_finite(
                selected_matrix_tensor[actual_query].sum(), label="attention row sum"
            ),
            "connections": connections,
            "top_predictions": prediction["predictions"],
            "probability_sum": prediction["probability_sum"],
            "attention_note": (
                "Attention patterns show where this head reads information from; they do not "
                "completely explain the model's prediction."
            ),
        }

    def predict(
        self,
        text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Return temperature-adjusted next-token probabilities for one prompt."""

        prompt = _validate_prompt(text)
        requested_top_k = _positive_integer(top_k, "top_k")
        selected_temperature = _validate_temperature(temperature)
        bundle = self._get_or_load_bundle()
        encoded = self._encode_prompt(bundle, prompt)

        with self._inference_lock, bundle.torch.no_grad():
            outputs = bundle.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise PretrainedCompatibilityError("The configured causal model did not return logits.")
        prediction = self._prediction_values(
            bundle,
            logits[0, -1],
            top_k=requested_top_k,
            temperature=selected_temperature,
        )
        return {
            "model_name": bundle.model_name,
            "device": bundle.metadata["device"],
            "input_text": prompt,
            "tokens": encoded["tokens"],
            "token_ids": encoded["token_ids"],
            "token_count": len(encoded["token_ids"]),
            "original_token_count": encoded["original_token_count"],
            "context_truncated": encoded["context_truncated"],
            "temperature": _rounded_finite(selected_temperature, label="temperature"),
            "top_k": requested_top_k,
            "predictions": prediction["predictions"],
            "probability_sum": prediction["probability_sum"],
            "top_probability_sum": prediction["top_probability_sum"],
            "probability_label": "model probability",
        }

    def _get_or_load_bundle(self) -> _ModelBundle:
        with self._condition:
            if self._bundle is not None:
                return self._bundle
        # Inspect and predict are also legitimate lazy-load entry points.  This
        # keeps the service usable if a client skips the explicit /load request.
        self.load()
        with self._condition:
            if self._bundle is None:  # Defensive: load either returns or raises.
                raise PretrainedLoadError("The pretrained model is not loaded.")
            return self._bundle

    def _load_bundle(self) -> _ModelBundle:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError) as exc:
            raise PretrainedDependencyError(
                "Optional pretrained-model support requires both 'torch' and 'transformers'. "
                "Install the backend requirements, then try again."
            ) from exc

        if self.device_name == "cpu":
            device = torch.device("cpu")
        elif self.device_name == "cuda":
            if not torch.cuda.is_available():
                raise PretrainedLoadError(
                    "PRETRAINED_DEVICE is 'cuda', but CUDA is not available. Use 'cpu' or "
                    "install a CUDA-compatible PyTorch build."
                )
            device = torch.device("cuda")
        else:
            raise PretrainedLoadError("PRETRAINED_DEVICE must be either 'cpu' or 'cuda'.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model, eager_requested = self._load_model_with_eager(
                AutoModelForCausalLM, self.model_name
            )
        except PretrainedServiceError:
            raise
        except Exception as exc:
            raise PretrainedLoadError(
                f"Could not load pretrained model '{self.model_name}'. Check internet access, "
                "the configured model name, and the local Hugging Face cache."
            ) from exc

        config = model.config
        config.output_attentions = True
        if hasattr(config, "_attn_implementation"):
            try:
                config._attn_implementation = "eager"
            except (AttributeError, TypeError, ValueError):
                pass
        set_attention = getattr(model, "set_attn_implementation", None)
        if callable(set_attention):
            try:
                set_attention("eager")
                eager_requested = True
            except (AttributeError, TypeError, ValueError, NotImplementedError):
                # The inspection call verifies that attentions are truly returned.
                pass

        model.to(device)
        model.eval()
        metadata = {
            "name": self.model_name,
            "device": str(device),
            "number_of_layers": _config_integer(config, "num_hidden_layers", "n_layer"),
            "number_of_heads": _config_integer(config, "num_attention_heads", "n_head"),
            "hidden_dimension": _config_integer(config, "hidden_size", "n_embd"),
            "vocabulary_size": _config_integer(config, "vocab_size") or len(tokenizer),
            "context_length": _context_limit(tokenizer, config),
            "attention_implementation": "eager" if eager_requested else "model_default",
        }
        return _ModelBundle(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            model_name=self.model_name,
            device=device,
            metadata=metadata,
        )

    @staticmethod
    def _load_model_with_eager(auto_model: Any, model_name: str) -> tuple[Any, bool]:
        """Prefer eager attention, with a compatibility path for older releases."""

        try:
            return auto_model.from_pretrained(
                model_name,
                attn_implementation="eager",
            ), True
        except (TypeError, ValueError) as exc:
            message = str(exc).lower()
            compatibility_markers = (
                "attn_implementation",
                "attention implementation",
                "unexpected keyword",
            )
            if not any(marker in message for marker in compatibility_markers):
                raise
            return auto_model.from_pretrained(model_name), False

    @staticmethod
    def _encode_prompt(bundle: _ModelBundle, text: str) -> dict[str, Any]:
        try:
            encoded = bundle.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
        except Exception as exc:
            raise PretrainedValidationError("The configured tokenizer could not encode the prompt.") from exc

        input_ids = encoded.get("input_ids")
        if input_ids is None or getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
            raise PretrainedCompatibilityError(
                "The configured tokenizer returned an unsupported input shape."
            )
        original_count = int(input_ids.shape[1])
        if original_count == 0:
            raise PretrainedValidationError("text did not produce any model tokens.")

        limit = int(bundle.metadata["context_length"])
        truncated = original_count > limit
        if truncated:
            input_ids = input_ids[:, -limit:]

        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = bundle.torch.ones_like(input_ids)
        elif truncated:
            attention_mask = attention_mask[:, -limit:]

        input_ids = input_ids.to(bundle.device)
        attention_mask = attention_mask.to(bundle.device)
        token_ids = [int(value) for value in input_ids[0].detach().to("cpu").tolist()]
        tokens = [_token_string(bundle.tokenizer, token_id) for token_id in token_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_ids": token_ids,
            "tokens": tokens,
            "original_token_count": original_count,
            "context_truncated": truncated,
        }

    @staticmethod
    def _prediction_values(
        bundle: _ModelBundle,
        logits: Any,
        *,
        top_k: int,
        temperature: float,
    ) -> dict[str, Any]:
        vocabulary_size = int(logits.shape[-1])
        if top_k > vocabulary_size:
            raise PretrainedValidationError(
                f"top_k must be between 1 and {vocabulary_size} for this model."
            )
        adjusted_logits = logits / temperature
        probabilities = bundle.torch.softmax(adjusted_logits, dim=-1)
        probability_sum = _rounded_finite(
            probabilities.sum(), label="probability sum"
        )
        values, indices = bundle.torch.topk(probabilities, k=top_k)
        predictions: list[dict[str, Any]] = []
        for probability, token_id_tensor in zip(values, indices, strict=True):
            token_id = int(token_id_tensor.item())
            predictions.append(
                {
                    "token": _token_string(bundle.tokenizer, token_id),
                    "decoded_token": _decoded_token(bundle.tokenizer, token_id),
                    "token_id": token_id,
                    "logit": _rounded_finite(
                        logits[token_id], label="logit"
                    ),
                    "temperature_adjusted_logit": _rounded_finite(
                        adjusted_logits[token_id], label="temperature-adjusted logit"
                    ),
                    "probability": _rounded_finite(
                        probability, label="model probability"
                    ),
                }
            )
        return {
            "predictions": predictions,
            "probability_sum": probability_sum,
            "top_probability_sum": _rounded_finite(
                values.sum(), label="top probability sum"
            ),
        }


# A single process-local cache is appropriate for the educational FastAPI app.
# It allocates no model resources until load/inspect/predict is called.
pretrained_service = PretrainedModelService()


def get_pretrained_status() -> dict[str, Any]:
    """Callable for ``GET /api/pretrained/status``."""

    return pretrained_service.status()


def load_pretrained_model() -> dict[str, Any]:
    """Callable for ``POST /api/pretrained/load``."""

    return pretrained_service.load()


def inspect_pretrained_model(
    text: str,
    *,
    layer: int = 0,
    head: int = 0,
    query_index: int | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Callable for ``POST /api/pretrained/inspect``."""

    return pretrained_service.inspect(
        text,
        layer=layer,
        head=head,
        query_index=query_index,
        top_k=top_k,
    )


def predict_pretrained_model(
    text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Callable for ``POST /api/pretrained/predict``."""

    return pretrained_service.predict(text, top_k=top_k, temperature=temperature)


__all__ = [
    "DEFAULT_PRETRAINED_MODEL",
    "DEVICE_ENV",
    "MAX_PROMPT_CHARACTERS",
    "MODEL_NAME_ENV",
    "PretrainedCompatibilityError",
    "PretrainedDependencyError",
    "PretrainedLoadError",
    "PretrainedModelService",
    "PretrainedServiceError",
    "PretrainedValidationError",
    "get_pretrained_status",
    "inspect_pretrained_model",
    "load_pretrained_model",
    "predict_pretrained_model",
    "pretrained_service",
]
