"""Transparent Phase 2 tokenization and embedding calculations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


VOCABULARY = ("<UNK>", "I", "love", "you", "music", "coding", "AI", "transformers", ".", "!")
EMBEDDING_DIMENSION = 4

# A deliberately small, explicit embedding table keeps this teaching project
# reproducible and makes every value easy to inspect.
TOKEN_EMBEDDINGS: tuple[tuple[float, ...], ...] = (
    (0.00, 0.00, 0.00, 0.00),
    (0.12, -0.31, 0.48, 0.07),
    (0.61, 0.22, -0.18, 0.35),
    (0.09, 0.54, 0.27, -0.42),
    (-0.24, 0.46, 0.38, 0.15),
    (0.51, -0.08, 0.63, -0.29),
    (0.72, 0.19, -0.36, 0.44),
    (0.33, -0.57, 0.21, 0.68),
    (-0.11, 0.06, -0.04, 0.13),
    (0.18, 0.12, -0.09, 0.26),
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CANONICAL_BY_CASEFOLD = {token.casefold(): token for token in VOCABULARY[1:]}
_TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}


@dataclass(frozen=True)
class TokenEmbeddingResult:
    token: str
    normalized: str
    token_id: int
    position: int
    token_embedding: list[float]
    position_embedding: list[float]
    combined_embedding: list[float]


def tokenize(text: str) -> list[str]:
    """Split text into word runs and individual punctuation tokens."""
    return _TOKEN_PATTERN.findall(text)


def normalize_token(token: str) -> str:
    """Return canonical vocabulary spelling or ``<UNK>``."""
    return _CANONICAL_BY_CASEFOLD.get(token.casefold(), "<UNK>")


def positional_embedding(position: int) -> list[float]:
    """Return the standard sinusoidal encoding for one zero-based position."""
    values: list[float] = []
    for dimension in range(EMBEDDING_DIMENSION):
        frequency = 10000 ** (2 * (dimension // 2) / EMBEDDING_DIMENSION)
        angle = position / frequency
        values.append(math.sin(angle) if dimension % 2 == 0 else math.cos(angle))
    return values


def embed_text(text: str) -> list[TokenEmbeddingResult]:
    results: list[TokenEmbeddingResult] = []
    for position, token in enumerate(tokenize(text)):
        normalized = normalize_token(token)
        token_id = _TOKEN_IDS[normalized]
        token_vector = list(TOKEN_EMBEDDINGS[token_id])
        position_vector = positional_embedding(position)
        combined = [token_value + position_value for token_value, position_value in zip(token_vector, position_vector)]
        results.append(
            TokenEmbeddingResult(
                token=token,
                normalized=normalized,
                token_id=token_id,
                position=position,
                token_embedding=token_vector,
                position_embedding=position_vector,
                combined_embedding=combined,
            )
        )
    return results
