"""Validated configuration objects for the locally trainable decoder model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _integer_in_range(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture settings for the tiny pre-normalization decoder.

    ``vocab_size`` is replaced with the corpus tokenizer size by the trainer.  A
    useful default is retained so the object is independently constructible in
    educational examples and unit tests.
    """

    vocab_size: int = 128
    context_length: int = 16
    d_model: int = 32
    number_of_heads: int = 4
    number_of_layers: int = 2
    feed_forward_dimension: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _integer_in_range(self.vocab_size, "vocab_size", 4, 100_000)
        _integer_in_range(self.context_length, "context_length", 2, 512)
        _integer_in_range(self.d_model, "d_model", 4, 1_024)
        _integer_in_range(self.number_of_heads, "number_of_heads", 1, 64)
        _integer_in_range(self.number_of_layers, "number_of_layers", 1, 24)
        _integer_in_range(self.feed_forward_dimension, "feed_forward_dimension", 4, 8_192)
        dropout = _finite_number(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be at least 0 and less than 1")
        if self.d_model % self.number_of_heads != 0:
            raise ValueError("d_model must be divisible by number_of_heads")

    @property
    def head_dimension(self) -> int:
        return self.d_model // self.number_of_heads

    # Short aliases make the class pleasant to use in model code while the
    # serialized names stay faithful to the educational UI wording.
    @property
    def num_heads(self) -> int:
        return self.number_of_heads

    @property
    def num_layers(self) -> int:
        return self.number_of_layers

    @property
    def ffn_dimension(self) -> int:
        return self.feed_forward_dimension

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        data = dict(values)
        aliases = {
            "num_heads": "number_of_heads",
            "num_layers": "number_of_layers",
            "ffn_dimension": "feed_forward_dimension",
        }
        for alias, canonical in aliases.items():
            if alias in data and canonical not in data:
                data[canonical] = data.pop(alias)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Safe, API-friendly defaults for local CPU-sized training jobs."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    validation_fraction: float = 0.2
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        _integer_in_range(self.epochs, "epochs", 1, 500)
        _integer_in_range(self.batch_size, "batch_size", 1, 256)
        learning_rate = _finite_number(self.learning_rate, "learning_rate")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be greater than 0 and at most 1")
        weight_decay = _finite_number(self.weight_decay, "weight_decay")
        if not 0.0 <= weight_decay <= 10.0:
            raise ValueError("weight_decay must be between 0 and 10")
        gradient_clip = _finite_number(self.gradient_clip, "gradient_clip")
        if not 0.0 < gradient_clip <= 100.0:
            raise ValueError("gradient_clip must be greater than 0 and at most 100")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not -(2**31) <= self.seed <= 2**31 - 1:
            raise ValueError("seed is outside the supported range")
        validation_fraction = _finite_number(self.validation_fraction, "validation_fraction")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be greater than 0 and less than 0.5")
        _integer_in_range(self.num_workers, "num_workers", 0, 4)
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        return cls(**dict(values))


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()

