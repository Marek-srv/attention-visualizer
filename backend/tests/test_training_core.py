from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from app.training.config import ModelConfig
from app.training.dataset import CausalLanguageModelDataset, collate_causal_batch
from app.training.model import TinyDecoderLanguageModel
from app.training.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, WordPunctuationTokenizer


LINES = ["I love you!", "I love music.", "transformers process sequences."]


def test_vocabulary_is_deterministic_and_special_ids_survive_save_load() -> None:
    first = WordPunctuationTokenizer.build_from_lines(LINES)
    second = WordPunctuationTokenizer.build_from_lines(reversed(LINES))
    assert first.id_to_token == second.id_to_token
    assert first.id_to_token[:4] == [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    path = Path("tests/_vocabulary_test.json")
    try:
        first.save_vocabulary(path)
        loaded = WordPunctuationTokenizer.load_vocabulary(path)
        assert loaded.token_to_id == first.token_to_id
        assert loaded.pad_token_id == 0
        assert loaded.unk_token_id == 1
        assert loaded.bos_token_id == 2
        assert loaded.eos_token_id == 3
        assert json.loads(path.read_text(encoding="utf-8"))["tokens"] == first.id_to_token
    finally:
        path.unlink(missing_ok=True)


def test_tokenizer_preserves_punctuation_unknowns_and_known_round_trip() -> None:
    tokenizer = WordPunctuationTokenizer.build_from_lines(LINES)
    assert tokenizer.tokenize("I love you!") == ["I", "love", "you", "!"]
    assert tokenizer.encode("mystery")[0] == tokenizer.unk_token_id
    ids = tokenizer.encode("I love music.", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == "I love music."


def test_dataset_inputs_and_shifted_targets_align_and_padding_is_marked() -> None:
    tokenizer = WordPunctuationTokenizer.build_from_lines(LINES)
    dataset = CausalLanguageModelDataset(["I love you", "I love"], tokenizer, context_length=8)
    first = dataset[0]
    expected = tokenizer.encode("I love you", add_bos=True, add_eos=True)
    assert first["input_ids"].tolist() == expected[:-1]
    assert first["target_ids"].tolist() == expected[1:]

    batch = collate_causal_batch([dataset[0], dataset[1]], tokenizer.pad_token_id)
    assert batch["input_ids"].shape == batch["target_ids"].shape
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["target_ids"][1, -1].item() == tokenizer.pad_token_id
    assert not batch["attention_mask"][1, -1].item()


def _small_model(vocabulary_size: int, pad_token_id: int) -> TinyDecoderLanguageModel:
    config = ModelConfig(
        vocab_size=vocabulary_size,
        context_length=8,
        d_model=8,
        number_of_heads=2,
        number_of_layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    return TinyDecoderLanguageModel(config, pad_token_id=pad_token_id)


def test_model_shapes_causality_trace_and_normal_forward() -> None:
    tokenizer = WordPunctuationTokenizer.build_from_lines(LINES)
    model = _small_model(tokenizer.vocabulary_size, tokenizer.pad_token_id).eval()
    input_ids = torch.tensor([tokenizer.encode("I love you", add_bos=True)], dtype=torch.long)

    with torch.no_grad():
        normal = model(input_ids)
        traced = model(input_ids, return_trace=True)
    assert normal.logits.shape == (1, input_ids.shape[1], tokenizer.vocabulary_size)
    assert normal.trace is None
    assert traced.trace is not None
    trace = traced.trace
    assert len(trace.layers) == 1
    layer = trace.layers[0]
    assert layer.query.shape == (1, 2, input_ids.shape[1], 4)
    assert layer.key.shape == layer.query.shape
    assert layer.value.shape == layer.query.shape
    assert layer.attention_residual_output.shape == (1, input_ids.shape[1], 8)
    assert layer.feed_forward_pre_activations.shape == (1, input_ids.shape[1], 16)
    assert layer.gelu_activations.shape == layer.feed_forward_pre_activations.shape
    assert layer.block_output.shape == (1, input_ids.shape[1], 8)
    assert trace.final_hidden_states.shape == (1, input_ids.shape[1], 8)
    assert trace.vocabulary_logits.shape == normal.logits.shape
    assert trace.vocabulary_probabilities.shape == normal.logits.shape
    probabilities = layer.attention_probabilities
    for query in range(input_ids.shape[1]):
        assert probabilities[0, :, query, query + 1 :].eq(0).all()
        assert torch.allclose(probabilities[0, :, query, :].sum(dim=-1), torch.ones(2), atol=1e-6)


def test_padding_positions_are_ignored_by_loss() -> None:
    tokenizer = WordPunctuationTokenizer.build_from_lines(LINES)
    model = _small_model(tokenizer.vocabulary_size, tokenizer.pad_token_id).eval()
    input_ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.token_id("I"), tokenizer.pad_token_id]])
    targets = torch.tensor([[tokenizer.token_id("I"), tokenizer.eos_token_id, tokenizer.pad_token_id]])
    with torch.no_grad():
        output = model(input_ids, targets=targets)
    expected = F.cross_entropy(
        output.logits.reshape(-1, tokenizer.vocabulary_size),
        targets.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.loss.item() == pytest.approx(expected.item())


def test_fixed_seed_initialization_is_reproducible() -> None:
    tokenizer = WordPunctuationTokenizer.build_from_lines(LINES)
    torch.manual_seed(42)
    first = _small_model(tokenizer.vocabulary_size, tokenizer.pad_token_id)
    torch.manual_seed(42)
    second = _small_model(tokenizer.vocabulary_size, tokenizer.pad_token_id)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)
    assert all(math.isfinite(value.item()) for parameter in first.parameters() for value in parameter.flatten())
