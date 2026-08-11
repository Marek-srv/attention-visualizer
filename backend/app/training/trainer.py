"""Finite, deterministic training loop and a single background-job manager."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable
from uuid import uuid4

import torch

from .checkpoint import CheckpointManager, DEFAULT_CHECKPOINT_FILENAME
from .config import ModelConfig, TrainingConfig
from .dataset import create_data_loaders, read_corpus_lines
from .inference import choose_device
from .model import TinyDecoderLanguageModel
from .tokenizer import WordPunctuationTokenizer


LOGGER = logging.getLogger(__name__)
TRAINING_STATES = {"idle", "running", "completed", "failed", "cancelled"}


class TrainingAlreadyRunningError(RuntimeError):
    pass


@dataclass(slots=True)
class TrainingResult:
    model: TinyDecoderLanguageModel
    tokenizer: WordPunctuationTokenizer
    model_config: ModelConfig
    training_config: TrainingConfig
    history: list[dict[str, Any]]
    best_validation_loss: float
    best_checkpoint: str | None
    cancelled: bool


def default_corpus_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "tiny_corpus.txt"


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only prevents a platform-specific unsupported kernel from turning a
    # small educational job into an opaque crash.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def safe_perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        raise RuntimeError("loss became non-finite")
    # exp(50) is already enormous and remains JSON-safe on every platform.
    return math.exp(min(loss, 50.0))


def _epoch_loss(
    model: TinyDecoderLanguageModel,
    loader: Any,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    accumulated_loss = 0.0
    token_count = 0
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs, targets=targets, attention_mask=attention_mask)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("training produced a non-finite loss")
            if optimizer is not None:
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            non_padding_tokens = int(targets.ne(model.pad_token_id).sum().item())
            accumulated_loss += float(output.loss.detach().cpu()) * non_padding_tokens
            token_count += non_padding_tokens
    if token_count == 0:
        raise RuntimeError("the dataset contains no non-padding target tokens")
    mean_loss = accumulated_loss / token_count
    if not math.isfinite(mean_loss):
        raise RuntimeError("training produced a non-finite mean loss")
    return mean_loss


def train_tiny_model(
    corpus_path: str | Path | None = None,
    *,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    cancellation_event: threading.Event | None = None,
    epoch_callback: Callable[[dict[str, Any], float], None] | None = None,
) -> TrainingResult:
    """Run a synchronous training job; the manager places this in a thread."""

    requested_model_config = model_config or ModelConfig()
    settings = training_config or TrainingConfig()
    source_path = Path(corpus_path) if corpus_path is not None else default_corpus_path()
    lines = read_corpus_lines(source_path)
    tokenizer = WordPunctuationTokenizer.build_from_lines(lines)
    effective_model_config = replace(requested_model_config, vocab_size=tokenizer.vocabulary_size)
    set_deterministic_seed(settings.seed)
    device = choose_device(settings.device)
    training_loader, validation_loader = create_data_loaders(
        lines,
        tokenizer,
        context_length=effective_model_config.context_length,
        batch_size=settings.batch_size,
        validation_fraction=settings.validation_fraction,
        seed=settings.seed,
        num_workers=settings.num_workers,
    )
    model = TinyDecoderLanguageModel(effective_model_config, pad_token_id=tokenizer.pad_token_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    manager = checkpoint_manager or CheckpointManager()
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_checkpoint: str | None = None
    cancelled = False
    cancellation = cancellation_event or threading.Event()

    for epoch in range(1, settings.epochs + 1):
        if cancellation.is_set():
            cancelled = True
            break
        started_at = time.perf_counter()
        training_loss = _epoch_loss(
            model,
            training_loader,
            device,
            optimizer=optimizer,
            gradient_clip=settings.gradient_clip,
        )
        validation_loss = _epoch_loss(
            model,
            validation_loader,
            device,
            optimizer=None,
            gradient_clip=settings.gradient_clip,
        )
        duration = time.perf_counter() - started_at
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_perplexity": safe_perplexity(training_loss),
            "validation_perplexity": safe_perplexity(validation_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": duration,
        }
        history.append(metrics)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            manager.save(
                model,
                tokenizer,
                effective_model_config,
                settings,
                epoch=epoch,
                history=history,
                validation_loss=validation_loss,
                optimizer=optimizer,
                filename=checkpoint_filename,
            )
            best_checkpoint = checkpoint_filename
        if epoch_callback is not None:
            epoch_callback(dict(metrics), best_validation_loss)
        if cancellation.is_set() and epoch < settings.epochs:
            cancelled = True
            break

    if not history and not cancelled:
        raise RuntimeError("training completed without producing metrics")
    if not math.isfinite(best_validation_loss):
        # Cancellation before epoch one is valid; no checkpoint exists yet.
        best_validation_loss = 0.0
    return TrainingResult(
        model=model,
        tokenizer=tokenizer,
        model_config=effective_model_config,
        training_config=settings,
        history=history,
        best_validation_loss=best_validation_loss,
        best_checkpoint=best_checkpoint,
        cancelled=cancelled,
    )


class TrainingManager:
    """Coordinates one cooperative local training thread at a time."""

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        *,
        checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    ) -> None:
        self.corpus_path = Path(corpus_path) if corpus_path is not None else default_corpus_path()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        # Validate once at construction; callers never provide arbitrary paths.
        self.checkpoint_filename = self.checkpoint_manager.checkpoint_path(checkpoint_filename).name
        self._lock = threading.RLock()
        self._cancellation = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = self._idle_state()

    def _idle_state(self) -> dict[str, Any]:
        return {
            "state": "idle",
            "status": "idle",
            "job_id": None,
            "current_epoch": 0,
            "total_epochs": 0,
            "latest_completed_epoch": 0,
            "latest_metrics": None,
            "history": [],
            "best_validation_loss": None,
            "cancellation_requested": False,
            "checkpoint_file": self.checkpoint_filename,
            "checkpoint_available": self.checkpoint_manager.checkpoint_path(self.checkpoint_filename).is_file(),
            "error": None,
        }

    def start(
        self,
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> dict[str, Any]:
        requested_model = model_config or ModelConfig()
        requested_training = training_config or TrainingConfig()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise TrainingAlreadyRunningError("a training job is already running")
            self._cancellation = threading.Event()
            job_id = uuid4().hex
            self._state = {
                "state": "running",
                "status": "running",
                "job_id": job_id,
                "current_epoch": 0,
                "total_epochs": requested_training.epochs,
                "latest_completed_epoch": 0,
                "latest_metrics": None,
                "history": [],
                "best_validation_loss": None,
                "cancellation_requested": False,
                "checkpoint_file": self.checkpoint_filename,
                "checkpoint_available": self.checkpoint_manager.checkpoint_path(self.checkpoint_filename).is_file(),
                "error": None,
                "model_config": requested_model.to_dict(),
                "training_config": requested_training.to_dict(),
            }
            self._thread = threading.Thread(
                target=self._run_job,
                args=(job_id, requested_model, requested_training),
                name=f"tiny-transformer-training-{job_id[:8]}",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def _run_job(
        self,
        job_id: str,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        def record_epoch(metrics: dict[str, Any], best_validation_loss: float) -> None:
            safe_metrics = {
                key: (_rounded_metric(value) if isinstance(value, float) else value)
                for key, value in metrics.items()
            }
            with self._lock:
                if self._state.get("job_id") != job_id:
                    return
                history = [*self._state["history"], safe_metrics]
                self._state.update(
                    current_epoch=metrics["epoch"],
                    latest_completed_epoch=metrics["epoch"],
                    latest_metrics=safe_metrics,
                    history=history,
                    best_validation_loss=_rounded_metric(best_validation_loss),
                    checkpoint_available=True,
                )

        try:
            result = train_tiny_model(
                self.corpus_path,
                model_config=model_config,
                training_config=training_config,
                checkpoint_manager=self.checkpoint_manager,
                checkpoint_filename=self.checkpoint_filename,
                cancellation_event=self._cancellation,
                epoch_callback=record_epoch,
            )
            with self._lock:
                if self._state.get("job_id") == job_id:
                    final_state = "cancelled" if result.cancelled else "completed"
                    self._state.update(
                        state=final_state,
                        status=final_state,
                        cancellation_requested=result.cancelled,
                        model_config=result.model_config.to_dict(),
                        training_config=result.training_config.to_dict(),
                        checkpoint_available=result.best_checkpoint is not None
                        or self.checkpoint_manager.checkpoint_path(self.checkpoint_filename).is_file(),
                    )
        except Exception:
            LOGGER.exception("Tiny Transformer training job %s failed", job_id)
            with self._lock:
                if self._state.get("job_id") == job_id:
                    self._state.update(
                        state="failed",
                        status="failed",
                        error="Training failed. Check the backend log for details.",
                    )

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._state["state"] == "running":
                self._cancellation.set()
                self._state["cancellation_requested"] = True
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._state)
            result["history"] = [dict(metric) for metric in self._state.get("history", [])]
            result["latest_metrics"] = (
                dict(self._state["latest_metrics"]) if self._state.get("latest_metrics") else None
            )
            return result

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.status()


def _rounded_metric(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError("training metric became non-finite")
    return round(float(value), 6)


# Alternate service name used by thin API route modules.
TrainingService = TrainingManager
