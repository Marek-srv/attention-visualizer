"""Safe checkpoint persistence for the tiny local language model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import ModelConfig, TrainingConfig
from .model import TinyDecoderLanguageModel
from .tokenizer import WordPunctuationTokenizer


DEFAULT_CHECKPOINT_FILENAME = "tiny_transformer_best.pt"
DEFAULT_METADATA_FILENAME = "tiny_transformer_best.json"
CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


class CheckpointNotFoundError(FileNotFoundError):
    pass


@dataclass(slots=True)
class LoadedCheckpoint:
    model: TinyDecoderLanguageModel
    tokenizer: WordPunctuationTokenizer
    model_config: ModelConfig
    training_config: TrainingConfig
    epoch: int
    history: list[dict[str, Any]]
    validation_loss: float
    random_seed: int
    optimizer_state_dict: dict[str, Any] | None
    metadata: dict[str, Any]


def default_checkpoint_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints"


def _safe_filename(filename: str, required_suffix: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("checkpoint filename must be a non-empty string")
    candidate = Path(filename)
    if candidate.is_absolute() or candidate.name != filename or any(part == ".." for part in candidate.parts):
        raise ValueError("checkpoint filename must not contain a path")
    if candidate.suffix.lower() != required_suffix:
        raise ValueError(f"checkpoint filename must end in {required_suffix}")
    return candidate.name


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CheckpointError(f"checkpoint {name} must be finite")
    return float(value)


class CheckpointManager:
    """Reads and writes only simple filenames beneath one configured root."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = (Path(directory) if directory is not None else default_checkpoint_directory()).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, filename: str = DEFAULT_CHECKPOINT_FILENAME) -> Path:
        safe_name = _safe_filename(filename, ".pt")
        candidate = (self.directory / safe_name).resolve()
        if candidate.parent != self.directory:
            raise ValueError("checkpoint path must stay inside the checkpoint directory")
        return candidate

    def metadata_path(self, checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME) -> Path:
        safe_name = _safe_filename(checkpoint_filename, ".pt")
        return self.directory / (Path(safe_name).stem + ".json")

    def save(
        self,
        model: TinyDecoderLanguageModel,
        tokenizer: WordPunctuationTokenizer,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        *,
        epoch: int,
        history: Sequence[Mapping[str, Any]],
        validation_loss: float,
        optimizer: torch.optim.Optimizer | None = None,
        filename: str = DEFAULT_CHECKPOINT_FILENAME,
    ) -> dict[str, Any]:
        checkpoint_path = self.checkpoint_path(filename)
        metadata_path = self.metadata_path(filename)
        finite_validation_loss = _finite_float(validation_loss, "validation_loss")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        serialized_history = [dict(metric) for metric in history]
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "vocabulary": tokenizer.to_dict(),
            "epoch": epoch,
            "history": serialized_history,
            "validation_loss": finite_validation_loss,
            "random_seed": training_config.seed,
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)

        metadata: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_file": checkpoint_path.name,
            "metadata_file": metadata_path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "validation_loss": finite_validation_loss,
            "random_seed": training_config.seed,
            "vocabulary_size": tokenizer.vocabulary_size,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "history": serialized_history,
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_metadata.replace(metadata_path)
        return metadata

    def load(
        self,
        filename: str = DEFAULT_CHECKPOINT_FILENAME,
        *,
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> LoadedCheckpoint:
        checkpoint_path = self.checkpoint_path(filename)
        if not checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"checkpoint '{checkpoint_path.name}' does not exist")
        try:
            payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:  # Compatibility with older supported PyTorch releases.
            payload = torch.load(checkpoint_path, map_location=device)
        except Exception as exception:
            raise CheckpointError("checkpoint could not be loaded") from exception
        if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError("checkpoint format is unsupported")
        try:
            model_config = ModelConfig.from_dict(payload["model_config"])
            training_config = TrainingConfig.from_dict(payload["training_config"])
            tokenizer = WordPunctuationTokenizer.from_dict(payload["vocabulary"])
            if model_config.vocab_size != tokenizer.vocabulary_size:
                raise CheckpointError("checkpoint vocabulary does not match its model configuration")
            model = TinyDecoderLanguageModel(model_config, pad_token_id=tokenizer.pad_token_id)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(device)
            model.eval()
            optimizer_state = payload.get("optimizer_state_dict")
            if optimizer is not None and optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            epoch = int(payload["epoch"])
            validation_loss = _finite_float(payload["validation_loss"], "validation_loss")
            history = [dict(metric) for metric in payload.get("history", [])]
            random_seed = int(payload.get("random_seed", training_config.seed))
        except CheckpointError:
            raise
        except Exception as exception:
            raise CheckpointError("checkpoint contents are invalid") from exception
        metadata = self.read_metadata(filename)
        return LoadedCheckpoint(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            training_config=training_config,
            epoch=epoch,
            history=history,
            validation_loss=validation_loss,
            random_seed=random_seed,
            optimizer_state_dict=optimizer_state,
            metadata=metadata,
        )

    def read_metadata(self, filename: str = DEFAULT_CHECKPOINT_FILENAME) -> dict[str, Any]:
        metadata_path = self.metadata_path(filename)
        if not metadata_path.is_file():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise CheckpointError("checkpoint metadata could not be read") from exception
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint metadata is invalid")
        return payload

    def status(self, filename: str = DEFAULT_CHECKPOINT_FILENAME) -> dict[str, Any]:
        checkpoint_path = self.checkpoint_path(filename)
        metadata_path = self.metadata_path(filename)
        metadata = self.read_metadata(filename) if metadata_path.is_file() else {}
        return {
            "available": checkpoint_path.is_file(),
            "loaded": False,
            "checkpoint_file": checkpoint_path.name,
            "metadata_file": metadata_path.name,
            "metadata": metadata,
        }


def save_checkpoint(*args: Any, directory: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    return CheckpointManager(directory).save(*args, **kwargs)


def load_checkpoint(
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
    *,
    directory: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    return CheckpointManager(directory).load(filename, device=device)

