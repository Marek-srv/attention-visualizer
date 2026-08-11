from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.trainable_routes as trainable_routes
from app.main import app
from app.training.checkpoint import CheckpointManager, CheckpointNotFoundError
from app.training.config import ModelConfig, TrainingConfig
from app.training.inference import InferenceValidationError, TinyModelService
from app.training.trainer import TrainingAlreadyRunningError, TrainingManager, train_tiny_model
from app.training.tokenizer import BOS_TOKEN, PAD_TOKEN


CORPUS_PATH = Path("data/tiny_corpus.txt")
ARTIFACT_DIRECTORY = Path("tests/_training_artifacts")


def _clean_artifacts() -> None:
    if ARTIFACT_DIRECTORY.is_dir():
        for path in ARTIFACT_DIRECTORY.iterdir():
            if path.is_file():
                path.unlink()
        ARTIFACT_DIRECTORY.rmdir()


@pytest.fixture(scope="module")
def trained_service() -> TinyModelService:
    _clean_artifacts()
    manager = CheckpointManager(ARTIFACT_DIRECTORY)
    model_config = ModelConfig(
        context_length=12,
        d_model=8,
        number_of_heads=2,
        number_of_layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        epochs=2,
        batch_size=32,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=1.0,
        seed=42,
    )
    result = train_tiny_model(
        CORPUS_PATH,
        model_config=model_config,
        training_config=training_config,
        checkpoint_manager=manager,
    )
    assert len(result.history) == 2
    assert all(math.isfinite(epoch["training_loss"]) for epoch in result.history)
    assert all(math.isfinite(epoch["validation_loss"]) for epoch in result.history)
    assert manager.status()["available"]
    assert manager.metadata_path().is_file()

    loaded = manager.load()
    assert loaded.epoch >= 1
    assert loaded.model_config.vocab_size == loaded.tokenizer.vocabulary_size
    service = TinyModelService(manager, device="cpu")
    service.load()
    yield service
    _clean_artifacts()


def test_prediction_probabilities_and_top_k(trained_service: TinyModelService) -> None:
    prediction = trained_service.predict("I love", top_k=3, temperature=1.0)
    assert prediction["tokens"][0] == "<BOS>"
    assert len(prediction["predictions"]) == 3
    assert prediction["probability_sum"] == pytest.approx(1.0, abs=0.000001)
    assert all(math.isfinite(item["probability"]) for item in prediction["predictions"])
    with pytest.raises(InferenceValidationError):
        trained_service.predict("I love", top_k=3, temperature=0)


def test_greedy_and_seeded_sample_generation_are_reproducible(trained_service: TinyModelService) -> None:
    first_greedy = trained_service.generate("I love", max_new_tokens=4, strategy="greedy", top_k=5)
    second_greedy = trained_service.generate("I love", max_new_tokens=4, strategy="greedy", top_k=5)
    assert first_greedy["steps"] == second_greedy["steps"]

    first_sample = trained_service.generate("I love", max_new_tokens=4, strategy="sample", top_k=5, seed=7)
    second_sample = trained_service.generate("I love", max_new_tokens=4, strategy="sample", top_k=5, seed=7)
    assert first_sample["steps"] == second_sample["steps"]
    assert len(first_sample["steps"]) <= 4
    assert first_sample["stop_reason"] in {"eos", "max_new_tokens"}
    if first_sample["stop_reason"] == "eos":
        assert first_sample["steps"][-1]["is_eos"] is True
    else:
        assert len(first_sample["steps"]) == 4
    assert all(step["chosen_token"] not in {PAD_TOKEN, BOS_TOKEN} for step in first_sample["steps"])


def test_trained_inspection_returns_selected_trace(trained_service: TinyModelService) -> None:
    inspection = trained_service.inspect(
        "I love", layer=0, head=0, query_token=2, key_token=1, hidden_dimension=0, top_k=3
    )
    assert inspection["selection"] == {
        "layer": 0,
        "head": 0,
        "query_token": 2,
        "key_token": 1,
        "hidden_dimension": 0,
    }
    assert inspection["shapes"]["query"][1] == 2
    assert inspection["selected_attention_calculation"]["causally_masked"] is False
    assert inspection["probability_sum"] == pytest.approx(1.0, abs=0.000001)


def test_missing_checkpoint_is_reported() -> None:
    artifact_directory = Path("tests/_missing_checkpoint_artifacts")
    manager = CheckpointManager(artifact_directory)
    try:
        with pytest.raises(CheckpointNotFoundError):
            manager.load()
    finally:
        artifact_directory.rmdir()


def test_missing_checkpoint_endpoint_returns_404_and_rejects_path_traversal() -> None:
    artifact_directory = Path("tests/_missing_checkpoint_api_artifacts")
    original_service = trainable_routes.model_service
    trainable_routes.model_service = TinyModelService(CheckpointManager(artifact_directory), device="cpu")
    client = TestClient(app)
    try:
        response = client.post("/api/model/load", json={})
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]
        traversal = client.post("/api/model/load", json={"checkpoint_name": "../outside.pt"})
        assert traversal.status_code == 422
        assert "must not contain a path" in traversal.json()["detail"]
    finally:
        trainable_routes.model_service = original_service
        artifact_directory.rmdir()


def test_background_status_exclusivity_and_cancellation() -> None:
    _clean_artifacts()
    manager = TrainingManager(CORPUS_PATH, CheckpointManager(ARTIFACT_DIRECTORY))
    model_config = ModelConfig(
        context_length=8,
        d_model=8,
        number_of_heads=2,
        number_of_layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    training_config = TrainingConfig(epochs=50, batch_size=32, learning_rate=0.01, seed=42)
    started = manager.start(model_config, training_config)
    assert started["status"] == "running"
    with pytest.raises(TrainingAlreadyRunningError):
        manager.start(model_config, training_config)
    manager.cancel()
    finished = manager.wait(timeout=30)
    assert finished["status"] == "cancelled"
    restarted = manager.start(
        model_config,
        TrainingConfig(epochs=1, batch_size=32, learning_rate=0.01, seed=42),
    )
    assert restarted["status"] == "running"
    completed = manager.wait(timeout=30)
    assert completed["status"] == "completed"
    assert completed["latest_completed_epoch"] == 1
    assert len(completed["history"]) == 1
    assert completed["checkpoint_available"] is True
    _clean_artifacts()


def test_training_api_rejects_second_simultaneous_job() -> None:
    _clean_artifacts()
    original_manager = trainable_routes.training_manager
    replacement = TrainingManager(CORPUS_PATH, CheckpointManager(ARTIFACT_DIRECTORY))
    trainable_routes.training_manager = replacement
    client = TestClient(app)
    payload = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.01,
        "model_config": {
            "context_length": 8,
            "d_model": 8,
            "number_of_heads": 2,
            "number_of_layers": 1,
            "feed_forward_dimension": 16,
            "dropout": 0.0,
        },
    }
    try:
        assert client.post("/api/training/start", json=payload).status_code == 200
        assert client.post("/api/training/start", json=payload).status_code == 409
        assert client.post("/api/training/cancel", json={}).status_code == 200
        assert replacement.wait(timeout=30)["status"] == "cancelled"
    finally:
        replacement.cancel()
        replacement.wait(timeout=30)
        trainable_routes.training_manager = original_manager
        _clean_artifacts()
