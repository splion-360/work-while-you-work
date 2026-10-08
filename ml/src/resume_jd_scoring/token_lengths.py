import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class PretrainedTokenizer(Protocol):
    name_or_path: str
    vocab_size: int
    model_max_length: int
    bos_token_id: int | None
    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool, **kwargs: Any) -> Mapping[str, Any]: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


@dataclass(frozen=True)
class HuggingFaceTokenCounter:
    tokenizer: PretrainedTokenizer

    @classmethod
    def from_pretrained(cls, model_path: Path | str) -> "HuggingFaceTokenCounter":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
        )
        return cls(tokenizer)

    def count(self, text: str) -> int:
        if not text:
            return 0
        encoded = self.tokenizer(text, add_special_tokens=True)
        return len(encoded["input_ids"])

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "huggingface_pretrained",
            "name_or_path": self.tokenizer.name_or_path,
            "tokenizer_class": type(self.tokenizer).__name__,
            "vocabulary_size": self.tokenizer.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
            "special_tokens_per_sequence": self.tokenizer.num_special_tokens_to_add(pair=False),
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def token_length_summary(lengths: Iterable[int], context_length: int) -> dict[str, Any]:
    values = list(lengths)
    overflow = [length for length in values if length > context_length]
    removed = [length - context_length for length in overflow]
    return {
        "documents": len(values),
        "minimum": min(values, default=0),
        "median": statistics.median(values) if values else 0,
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "maximum": max(values, default=0),
        "over_context": len(overflow),
        "over_context_rate": round(len(overflow) / len(values), 6) if values else 0,
        "removed_tokens": {
            "minimum": min(removed, default=0),
            "median": statistics.median(removed) if removed else 0,
            "p95": nearest_rank(removed, 0.95),
            "maximum": max(removed, default=0),
        },
    }


def count_documents(
    documents: Mapping[str, str], counter: HuggingFaceTokenCounter
) -> dict[str, int]:
    return {item_hash: counter.count(text) for item_hash, text in sorted(documents.items())}
