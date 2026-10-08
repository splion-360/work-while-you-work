import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

DISTRIBUTION_STATISTICS = (
    "mean",
    "standard_deviation",
    "minimum",
    "q25",
    "median",
    "q75",
    "maximum",
)


@dataclass(frozen=True)
class KernelBank:
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    scale: float = 0.01

    def __post_init__(self) -> None:
        if not self.means or len(self.means) != len(self.standard_deviations):
            raise ValueError("kernel means and standard deviations must be non-empty and aligned")
        if any(value <= 0 for value in self.standard_deviations):
            raise ValueError("kernel standard deviations must be positive")
        if self.scale <= 0:
            raise ValueError("kernel scale must be positive")

    @property
    def count(self) -> int:
        return len(self.means)


def standard_knrm_kernels(count: int = 11) -> KernelBank:
    if count < 2:
        raise ValueError("K-NRM requires at least two kernels")
    bin_size = 2.0 / (count - 1)
    means = [1.0, 1.0 - bin_size / 2.0]
    means.extend(1.0 - bin_size / 2.0 - bin_size * index for index in range(1, count - 1))
    return KernelBank(
        means=tuple(means),
        standard_deviations=(0.001, *((0.1,) * (count - 1))),
    )


class TokenVectorCache:
    def __init__(self, vectors_path: Path, index_path: Path) -> None:
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.ndim != 2:
            raise ValueError("token vector cache must be a matrix")
        index_rows = pq.read_table(index_path).to_pylist()
        self.vectors = vectors
        self.index = {
            str(row["document_hash"]): (int(row["offset"]), int(row["token_count"]))
            for row in index_rows
        }
        if len(self.index) != len(index_rows):
            raise ValueError("token cache index contains duplicate document hashes")
        for document_hash, (offset, token_count) in self.index.items():
            if offset < 0 or token_count <= 0 or offset + token_count > len(vectors):
                raise ValueError(f"invalid token cache range for {document_hash}")

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def get(self, document_hash: str) -> np.ndarray:
        try:
            offset, token_count = self.index[document_hash]
        except KeyError as error:
            raise KeyError(f"document is absent from token cache: {document_hash}") from error
        return self.vectors[offset : offset + token_count]


def knrm_pool_similarity(
    jd_vectors: np.ndarray | torch.Tensor,
    resume_vectors: np.ndarray | torch.Tensor,
    kernels: KernelBank,
    *,
    device: str,
    resume_block_size: int = 256,
) -> np.ndarray:
    return knrm_pool_similarity_batch(
        jd_vectors,
        [resume_vectors],
        kernels,
        device=device,
        resume_block_size=resume_block_size,
    )[0]


def knrm_pool_similarity_batch(
    jd_vectors: np.ndarray | torch.Tensor,
    resume_vectors: list[np.ndarray | torch.Tensor],
    kernels: KernelBank,
    *,
    device: str,
    resume_block_size: int = 4096,
) -> np.ndarray:
    if resume_block_size <= 0:
        raise ValueError("resume block size must be positive")
    if not resume_vectors:
        raise ValueError("at least one resume token matrix is required")
    jd_array = np.array(jd_vectors, dtype=np.float32, copy=True)
    resume_arrays = [np.array(value, dtype=np.float32, copy=True) for value in resume_vectors]
    if jd_array.ndim != 2 or not jd_array.shape[0]:
        raise ValueError("JD token vectors must be a non-empty matrix")
    if any(
        value.ndim != 2 or not value.shape[0] or value.shape[1] != jd_array.shape[1]
        for value in resume_arrays
    ):
        raise ValueError("JD and resume token vectors must be aligned non-empty matrices")

    jd = torch.from_numpy(jd_array).to(device)
    resume = torch.from_numpy(np.concatenate(resume_arrays, axis=0))
    segment_ids = torch.repeat_interleave(
        torch.arange(len(resume_arrays), dtype=torch.int64),
        torch.tensor([len(value) for value in resume_arrays], dtype=torch.int64),
    )

    means = torch.tensor(kernels.means, dtype=torch.float32, device=device)
    variances = 2.0 * torch.tensor(
        kernels.standard_deviations, dtype=torch.float32, device=device
    ).square()
    soft_counts = torch.zeros(
        (jd.shape[0], len(resume_arrays), kernels.count), dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        for start in range(0, resume.shape[0], resume_block_size):
            block = resume[start : start + resume_block_size].to(device)
            block_segments = segment_ids[start : start + resume_block_size].to(device)
            similarities = jd @ block.T
            for kernel_index in range(kernels.count):
                activations = torch.exp(
                    -((similarities - means[kernel_index]).square()) / variances[kernel_index]
                )
                soft_counts[:, :, kernel_index].scatter_add_(
                    1,
                    block_segments.expand(jd.shape[0], -1),
                    activations,
                )
        pooled = torch.log(torch.clamp_min(soft_counts, 1e-10))
        features = pooled.sum(dim=0) * kernels.scale
    result = features.cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (len(resume_arrays), kernels.count) or not np.isfinite(result).all():
        raise ValueError("K-NRM pooling produced invalid features")
    return result


def distributional_knrm_pool_similarity(
    jd_vectors: np.ndarray | torch.Tensor,
    resume_vectors: np.ndarray | torch.Tensor,
    kernels: KernelBank,
    *,
    device: str,
    resume_block_size: int = 256,
) -> np.ndarray:
    return distributional_knrm_pool_similarity_batch(
        jd_vectors,
        [resume_vectors],
        kernels,
        device=device,
        resume_block_size=resume_block_size,
    )[0]


def distributional_knrm_pool_similarity_batch(
    jd_vectors: np.ndarray | torch.Tensor,
    resume_vectors: list[np.ndarray | torch.Tensor],
    kernels: KernelBank,
    *,
    device: str,
    resume_block_size: int = 4096,
) -> np.ndarray:
    """Preserve the distribution of length-normalized kernel responses over JD tokens."""
    if resume_block_size <= 0:
        raise ValueError("resume block size must be positive")
    if not resume_vectors:
        raise ValueError("at least one resume token matrix is required")
    jd_array = np.array(jd_vectors, dtype=np.float32, copy=True)
    resume_arrays = [np.array(value, dtype=np.float32, copy=True) for value in resume_vectors]
    if jd_array.ndim != 2 or not jd_array.shape[0]:
        raise ValueError("JD token vectors must be a non-empty matrix")
    if any(
        value.ndim != 2 or not value.shape[0] or value.shape[1] != jd_array.shape[1]
        for value in resume_arrays
    ):
        raise ValueError("JD and resume token vectors must be aligned non-empty matrices")

    jd = torch.from_numpy(jd_array).to(device)
    resume = torch.from_numpy(np.concatenate(resume_arrays, axis=0))
    resume_lengths = torch.tensor(
        [len(value) for value in resume_arrays], dtype=torch.float32, device=device
    )
    segment_ids = torch.repeat_interleave(
        torch.arange(len(resume_arrays), dtype=torch.int64),
        resume_lengths.to(dtype=torch.int64).cpu(),
    )
    means = torch.tensor(kernels.means, dtype=torch.float32, device=device)
    variances = 2.0 * torch.tensor(
        kernels.standard_deviations, dtype=torch.float32, device=device
    ).square()
    soft_counts = torch.zeros(
        (jd.shape[0], len(resume_arrays), kernels.count), dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        for start in range(0, resume.shape[0], resume_block_size):
            block = resume[start : start + resume_block_size].to(device)
            block_segments = segment_ids[start : start + resume_block_size].to(device)
            similarities = jd @ block.T
            for kernel_index in range(kernels.count):
                activations = torch.exp(
                    -((similarities - means[kernel_index]).square()) / variances[kernel_index]
                )
                soft_counts[:, :, kernel_index].scatter_add_(
                    1,
                    block_segments.expand(jd.shape[0], -1),
                    activations,
                )
        token_responses = torch.log(
            torch.clamp_min(soft_counts / resume_lengths[None, :, None], 1e-10)
        )
        statistics = torch.stack(
            (
                token_responses.mean(dim=0),
                token_responses.std(dim=0, unbiased=False),
                token_responses.amin(dim=0),
                torch.quantile(token_responses, 0.25, dim=0),
                torch.quantile(token_responses, 0.50, dim=0),
                torch.quantile(token_responses, 0.75, dim=0),
                token_responses.amax(dim=0),
            ),
            dim=2,
        )
        features = statistics.flatten(start_dim=1) * kernels.scale
    result = features.cpu().numpy().astype(np.float32, copy=False)
    expected_shape = (len(resume_arrays), kernels.count * len(DISTRIBUTION_STATISTICS))
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise ValueError("distributional K-NRM pooling produced invalid features")
    return result


def validate_normalized_token_vectors(vectors: np.ndarray, *, tolerance: float = 2e-3) -> None:
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError("token vectors must be a non-empty matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("token vectors contain non-finite values")
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if not np.allclose(norms, 1.0, atol=tolerance):
        maximum_error = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(f"token vectors are not normalized; maximum norm error={maximum_error}")


def token_cache_bytes(token_count: int, dimension: int, dtype: np.dtype) -> int:
    if token_count <= 0 or dimension <= 0:
        raise ValueError("token count and dimension must be positive")
    return math.prod((token_count, dimension)) * np.dtype(dtype).itemsize
