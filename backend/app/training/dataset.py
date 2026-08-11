"""Causal next-token datasets and deterministic train/validation loaders."""

from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import WordPunctuationTokenizer


class CausalExample(TypedDict):
    input_ids: torch.Tensor
    target_ids: torch.Tensor


class CausalBatch(TypedDict):
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    attention_mask: torch.Tensor


class CausalLanguageModelDataset(Dataset[CausalExample]):
    """One shifted causal-LM example per non-empty corpus line.

    A line ``I love you`` becomes ``<BOS> I love you`` as the input and
    ``I love you <EOS>`` as the target.  Long lines are split into contiguous
    context-sized windows so no token disappears from training.
    """

    def __init__(
        self,
        lines: Iterable[str],
        tokenizer: WordPunctuationTokenizer,
        context_length: int = 16,
    ) -> None:
        if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 2:
            raise ValueError("context_length must be an integer of at least 2")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples: list[tuple[list[int], list[int]]] = []
        for line in lines:
            if not isinstance(line, str) or not line.strip():
                continue
            complete = tokenizer.encode(line.strip(), add_bos=True, add_eos=True)
            inputs = complete[:-1]
            targets = complete[1:]
            for start in range(0, len(inputs), context_length):
                input_window = inputs[start : start + context_length]
                target_window = targets[start : start + context_length]
                if input_window:
                    self.examples.append((input_window, target_window))
        if not self.examples:
            raise ValueError("the dataset requires at least one non-empty tokenized line")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CausalExample:
        inputs, targets = self.examples[index]
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "target_ids": torch.tensor(targets, dtype=torch.long),
        }


def collate_causal_batch(
    examples: Sequence[CausalExample],
    pad_token_id: int,
) -> CausalBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    maximum_length = max(example["input_ids"].numel() for example in examples)
    batch_size = len(examples)
    inputs = torch.full((batch_size, maximum_length), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, maximum_length), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch_size, maximum_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = example["input_ids"].numel()
        if example["target_ids"].numel() != length:
            raise ValueError("input and target sequences must have equal lengths")
        inputs[row, :length] = example["input_ids"]
        targets[row, :length] = example["target_ids"]
        mask[row, :length] = True
    return {"input_ids": inputs, "target_ids": targets, "attention_mask": mask}


def read_corpus_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_corpus_lines(
    lines: Sequence[str],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if len(lines) < 2:
        raise ValueError("at least two corpus lines are required for a train/validation split")
    indices = list(range(len(lines)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(len(lines) - 1, round(len(lines) * validation_fraction)))
    validation_indices = set(indices[:validation_count])
    training = [line for index, line in enumerate(lines) if index not in validation_indices]
    validation = [line for index, line in enumerate(lines) if index in validation_indices]
    return training, validation


def create_data_loaders(
    lines: Sequence[str],
    tokenizer: WordPunctuationTokenizer,
    *,
    context_length: int,
    batch_size: int,
    validation_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader[CausalBatch], DataLoader[CausalBatch]]:
    training_lines, validation_lines = split_corpus_lines(lines, validation_fraction, seed)
    training_dataset = CausalLanguageModelDataset(training_lines, tokenizer, context_length)
    validation_dataset = CausalLanguageModelDataset(validation_lines, tokenizer, context_length)
    generator = torch.Generator().manual_seed(seed)
    collate = partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    return training_loader, validation_loader


# Backward-friendly alias for tests and route integrations.
causal_lm_collate = collate_causal_batch
