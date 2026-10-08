from dataclasses import dataclass

from resume_jd_scoring.token_lengths import (
    HuggingFaceTokenCounter,
    count_documents,
    nearest_rank,
    token_length_summary,
)


@dataclass
class WordTokenizer:
    name_or_path: str = "test-tokenizer"
    vocab_size: int = 100
    model_max_length: int = 8
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __call__(self, text: str, *, add_special_tokens: bool, **_kwargs):
        token_ids = list(range(len(text.split())))
        if add_special_tokens and token_ids:
            token_ids = [self.bos_token_id, *token_ids, self.eos_token_id]
        return {"input_ids": token_ids}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert not pair
        return 2


def counter() -> HuggingFaceTokenCounter:
    return HuggingFaceTokenCounter(WordTokenizer())


def test_counter_uses_official_special_token_behavior():
    assert counter().count("one two three") == 5
    assert counter().count("") == 0


def test_counter_metadata_records_tokenizer_provenance():
    assert counter().metadata() == {
        "source": "huggingface_pretrained",
        "name_or_path": "test-tokenizer",
        "tokenizer_class": "WordTokenizer",
        "vocabulary_size": 100,
        "model_max_length": 8,
        "special_tokens_per_sequence": 2,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def test_nearest_rank_uses_observed_values():
    assert nearest_rank([10, 20, 30, 40], 0.75) == 30


def test_token_length_summary_reports_overflow():
    report = token_length_summary([100, 200, 300, 2200, 3000], context_length=2048)

    assert report["median"] == 300
    assert report["over_context"] == 2
    assert report["over_context_rate"] == 0.4
    assert report["removed_tokens"]["maximum"] == 952


def test_count_documents_is_hash_sorted():
    documents = {"b": "one two", "a": "one"}

    assert count_documents(documents, counter()) == {"a": 3, "b": 4}
