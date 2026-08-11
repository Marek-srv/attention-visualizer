"""A small, inspectable word-and-punctuation tokenizer."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)

# Words (including simple apostrophe forms) and every non-whitespace
# punctuation character become separate, visible tokens.
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['_’][^\W_]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


class WordPunctuationTokenizer:
    """Deterministic corpus tokenizer with an explicitly stored vocabulary."""

    def __init__(self, vocabulary: Sequence[str] | Mapping[str, int] | None = None) -> None:
        if vocabulary is None:
            ordered = list(SPECIAL_TOKENS)
        elif isinstance(vocabulary, Mapping):
            ordered_pairs = sorted(vocabulary.items(), key=lambda item: item[1])
            if [index for _, index in ordered_pairs] != list(range(len(ordered_pairs))):
                raise ValueError("vocabulary IDs must be contiguous and start at zero")
            ordered = [token for token, _ in ordered_pairs]
        else:
            ordered = list(vocabulary)
        self._validate_vocabulary(ordered)
        self.id_to_token: list[str] = ordered
        self.token_to_id: dict[str, int] = {token: index for index, token in enumerate(ordered)}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return TOKEN_PATTERN.findall(text)

    @classmethod
    def build_from_lines(
        cls,
        lines: Iterable[str],
        *,
        minimum_frequency: int = 1,
        max_vocabulary_size: int | None = None,
    ) -> "WordPunctuationTokenizer":
        if isinstance(minimum_frequency, bool) or minimum_frequency < 1:
            raise ValueError("minimum_frequency must be at least 1")
        if max_vocabulary_size is not None and max_vocabulary_size < len(SPECIAL_TOKENS):
            raise ValueError("max_vocabulary_size must leave room for all special tokens")
        counts: Counter[str] = Counter()
        for line in lines:
            counts.update(token for token in cls.tokenize(line) if token not in SPECIAL_TOKENS)
        # Frequency first, followed by Unicode lexical order, makes ties stable
        # across operating systems and repeated runs.
        corpus_tokens = sorted(
            (token for token, count in counts.items() if count >= minimum_frequency),
            key=lambda token: (-counts[token], token),
        )
        if max_vocabulary_size is not None:
            corpus_tokens = corpus_tokens[: max_vocabulary_size - len(SPECIAL_TOKENS)]
        return cls([*SPECIAL_TOKENS, *corpus_tokens])

    @classmethod
    def build_from_text(cls, text: str, **kwargs: object) -> "WordPunctuationTokenizer":
        return cls.build_from_lines(text.splitlines(), **kwargs)

    @classmethod
    def build_from_file(cls, path: str | Path, **kwargs: object) -> "WordPunctuationTokenizer":
        corpus_path = Path(path)
        return cls.build_from_lines(corpus_path.read_text(encoding="utf-8").splitlines(), **kwargs)

    @property
    def vocabulary_size(self) -> int:
        return len(self.id_to_token)

    @property
    def vocab_size(self) -> int:
        return self.vocabulary_size

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        if add_special_tokens is not None:
            add_bos = add_special_tokens
            add_eos = add_special_tokens
        ids = [self.token_id(token) for token in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def convert_ids_to_tokens(self, token_ids: Iterable[int]) -> list[str]:
        tokens: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            tokens.append(self.id_to_token[index] if 0 <= index < self.vocabulary_size else UNK_TOKEN)
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in SPECIAL_TOKENS]
        return self.detokenize(tokens)

    @staticmethod
    def detokenize(tokens: Sequence[str]) -> str:
        if not tokens:
            return ""
        no_space_before = set(".,!?;:%)]}»”’")
        no_space_after = set("([{«“‘")
        result = ""
        previous = ""
        for token in tokens:
            if not result:
                result = token
            elif token[0] in no_space_before or previous[-1:] in no_space_after:
                result += token
            else:
                result += " " + token
            previous = token
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "tokenizer_type": "word_and_punctuation",
            "special_tokens": list(SPECIAL_TOKENS),
            "tokens": list(self.id_to_token),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WordPunctuationTokenizer":
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("saved vocabulary must contain a string token list")
        return cls(tokens)

    def save_vocabulary(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        return output_path

    @classmethod
    def load_vocabulary(cls, path: str | Path) -> "WordPunctuationTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("saved vocabulary must be a JSON object")
        return cls.from_dict(payload)

    @staticmethod
    def _validate_vocabulary(tokens: Sequence[str]) -> None:
        if len(tokens) < len(SPECIAL_TOKENS):
            raise ValueError("vocabulary is missing required special tokens")
        if list(tokens[: len(SPECIAL_TOKENS)]) != list(SPECIAL_TOKENS):
            raise ValueError("special-token IDs must be PAD=0, UNK=1, BOS=2, EOS=3")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary tokens must be unique")
        if not all(isinstance(token, str) and token for token in tokens):
            raise ValueError("vocabulary tokens must be non-empty strings")


# Concise alias used by services and friendly to external callers.
TinyTokenizer = WordPunctuationTokenizer

