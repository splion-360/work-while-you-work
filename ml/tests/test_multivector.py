from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from resume_jd_scoring.multivector import (
    DISTRIBUTION_STATISTICS,
    TokenVectorCache,
    distributional_knrm_pool_similarity,
    distributional_knrm_pool_similarity_batch,
    knrm_pool_similarity,
    knrm_pool_similarity_batch,
    standard_knrm_kernels,
    token_cache_bytes,
    validate_normalized_token_vectors,
)


def test_standard_knrm_kernel_layout():
    kernels = standard_knrm_kernels()

    assert kernels.count == 11
    assert kernels.means == pytest.approx((1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9))
    assert kernels.standard_deviations[0] == 0.001
    assert kernels.standard_deviations[1:] == pytest.approx((0.1,) * 10)


def test_knrm_pooling_uses_all_similarity_levels_and_is_block_invariant():
    jd = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    resume = np.asarray([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]], dtype=np.float32)
    kernels = standard_knrm_kernels()

    unblocked = knrm_pool_similarity(jd, resume, kernels, device="cpu", resume_block_size=8)
    blocked = knrm_pool_similarity(jd, resume, kernels, device="cpu", resume_block_size=1)

    np.testing.assert_allclose(blocked, unblocked, rtol=1e-6, atol=1e-6)
    assert unblocked.shape == (11,)
    assert np.count_nonzero(np.abs(unblocked) > 1e-6) > 1


def test_batched_knrm_pooling_matches_individual_pooling():
    jd = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    resumes = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
    ]
    kernels = standard_knrm_kernels()

    batched = knrm_pool_similarity_batch(
        jd, resumes, kernels, device="cpu", resume_block_size=2
    )
    individual = np.stack(
        [
            knrm_pool_similarity(jd, resume, kernels, device="cpu", resume_block_size=2)
            for resume in resumes
        ]
    )

    np.testing.assert_allclose(batched, individual, rtol=1e-6, atol=1e-6)


def test_distributional_pooling_is_block_and_resume_length_invariant():
    jd = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    resume = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doubled_resume = np.repeat(resume, 2, axis=0)
    kernels = standard_knrm_kernels()

    base = distributional_knrm_pool_similarity(
        jd, resume, kernels, device="cpu", resume_block_size=8
    )
    blocked = distributional_knrm_pool_similarity(
        jd, resume, kernels, device="cpu", resume_block_size=1
    )
    doubled = distributional_knrm_pool_similarity(
        jd, doubled_resume, kernels, device="cpu", resume_block_size=2
    )

    assert base.shape == (kernels.count * len(DISTRIBUTION_STATISTICS),)
    np.testing.assert_allclose(blocked, base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(doubled, base, rtol=1e-6, atol=1e-6)


def test_batched_distributional_pooling_matches_individual_pooling():
    jd = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    resumes = [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
    ]
    kernels = standard_knrm_kernels()

    batched = distributional_knrm_pool_similarity_batch(
        jd, resumes, kernels, device="cpu", resume_block_size=2
    )
    individual = np.stack(
        [
            distributional_knrm_pool_similarity(
                jd, resume, kernels, device="cpu", resume_block_size=2
            )
            for resume in resumes
        ]
    )

    np.testing.assert_allclose(batched, individual, rtol=1e-6, atol=1e-6)


def test_token_vector_cache_reads_ragged_document_ranges(tmp_path: Path):
    vectors = np.arange(15, dtype=np.float16).reshape(5, 3)
    vectors_path = tmp_path / "vectors.npy"
    np.save(vectors_path, vectors)
    index_path = tmp_path / "index.parquet"
    pq.write_table(
        pa.table(
            {
                "document_hash": ["a", "b"],
                "offset": pa.array([0, 2], type=pa.int64()),
                "token_count": pa.array([2, 3], type=pa.int32()),
            }
        ),
        index_path,
    )

    cache = TokenVectorCache(vectors_path, index_path)

    assert cache.dimension == 3
    np.testing.assert_array_equal(cache.get("a"), vectors[:2])
    np.testing.assert_array_equal(cache.get("b"), vectors[2:])
    with pytest.raises(KeyError, match="absent"):
        cache.get("missing")


def test_token_vector_validation_and_size():
    vectors = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float16)

    validate_normalized_token_vectors(vectors)

    assert token_cache_bytes(5, 3, np.dtype("float16")) == 30
    with pytest.raises(ValueError, match="not normalized"):
        validate_normalized_token_vectors(np.asarray([[2.0, 0.0]], dtype=np.float32))
