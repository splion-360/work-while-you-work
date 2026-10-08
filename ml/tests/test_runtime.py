from pathlib import Path
from unittest.mock import patch

import pytest

from resume_jd_scoring.runtime import encoder_contract, verify_encoder


def test_encoder_contract_records_the_inference_representation():
    assert encoder_contract(8192) == {
        "pipeline_version": 1,
        "model_name": "BAAI/bge-m3",
        "representation": "normalized_colbert_token_vectors",
        "max_length": 8192,
        "cache_dtype": "float16",
        "excluded_token": "cls",
    }


def test_verify_encoder_rejects_files_that_do_not_match_manifest(tmp_path: Path):
    manifest = {"encoder": {"max_length": 8192, "fingerprint": "expected"}}
    with (
        patch("resume_jd_scoring.runtime.model_fingerprint", return_value=("actual", {})),
        pytest.raises(RuntimeError, match="do not match"),
    ):
        verify_encoder(tmp_path, manifest)
