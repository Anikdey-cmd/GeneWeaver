
"""CUDA mismatch-counting tests for GeneWeaver."""

import numpy as np
import pytest

# Skip these tests if Numba is not installed.
pytest.importorskip("numba")

from numba import cuda

# Require either a CUDA-capable GPU or CUDA simulation.
if not cuda.is_available():
    pytest.skip(
        "CUDA tests require an NVIDIA GPU or CUDA simulator.",
        allow_module_level=True,
    )

from src.cpu.cpu_alignment import count_mismatches
from src.gpu.cuda_alignment import count_mismatches_kernel


DNA_CODES = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
}


def encode_sequence(sequence: str) -> np.ndarray:
    """Convert DNA nucleotides into numeric codes."""
    return np.array(
        [DNA_CODES[base] for base in sequence],
        dtype=np.int32,
    )


def run_cuda_mismatch_test(guide, candidates):
    """Execute CUDA mismatch counting and return the results."""

    guide_codes = encode_sequence(guide)

    candidate_codes = np.stack([
        encode_sequence(candidate)
        for candidate in candidates
    ])

    mismatch_counts = np.zeros(
        len(candidates),
        dtype=np.int32,
    )

    threads_per_block = 32

    blocks_per_grid = (
        len(candidates) + threads_per_block - 1
    ) // threads_per_block

    count_mismatches_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        guide_codes,
        candidate_codes,
        mismatch_counts,
    )

    cuda.synchronize()

    return mismatch_counts.tolist()


def test_cuda_identical_sequences():
    guide = "A" * 20
    candidates = ["A" * 20]

    assert run_cuda_mismatch_test(guide, candidates) == [0]


def test_cuda_multiple_mismatch_counts():
    guide = "A" * 20

    candidates = [
        "A" * 20,
        "C" + "A" * 19,
        "CCC" + "A" * 17,
        "T" * 20,
    ]

    assert run_cuda_mismatch_test(guide, candidates) == [
        0, 1, 3, 20
    ]


def test_cuda_ambiguous_nucleotide():
    guide = "N" + "A" * 19

    candidates = [
        "N" + "A" * 19,
        "A" * 20,
        "C" + "A" * 19,
    ]

    assert run_cuda_mismatch_test(guide, candidates) == [
        1, 1, 1
    ]


@pytest.mark.parametrize(
    "guide,candidates",
    [
        ("ATGC" * 5, ["ATGC" * 5]),
        ("A" * 20, ["CCC" + "A" * 17]),
        ("N" + "A" * 19, ["N" + "A" * 19]),
    ],
)
def test_cuda_matches_cpu(guide, candidates):
    """Compare CUDA results against the CPU implementation."""

    gpu_results = run_cuda_mismatch_test(guide, candidates)

    cpu_results = [
        count_mismatches(guide, candidate)
        for candidate in candidates
    ]

    assert gpu_results == cpu_results
