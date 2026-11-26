
"""Tests for GeneWeaver GPU alignment and CPU fallback."""

import numpy as np
import pytest

from src.cpu.cpu_alignment import align_guide_to_candidates
from src.gpu.gpu_alignment import (
    encode_dna,
    align_guide_to_candidates_gpu,
)


GUIDE = "A" * 20


# -----------------------------------------
# DNA encoding tests
# -----------------------------------------

def test_encode_dna():
    encoded = encode_dna("ATGCN")

    assert encoded.tolist() == [0, 1, 2, 3, 4]
    assert encoded.dtype == np.int32


def test_encode_lowercase_dna():
    encoded = encode_dna("atg cn")

    assert encoded.tolist() == [0, 1, 2, 3, 4]


def test_encode_invalid_dna():
    with pytest.raises(ValueError):
        encode_dna("ATGX")


def test_encode_invalid_type():
    with pytest.raises(TypeError):
        encode_dna(123)


# -----------------------------------------
# Input validation tests
# -----------------------------------------

def test_empty_candidates():
    results = align_guide_to_candidates_gpu(GUIDE, [])

    assert results == []


def test_invalid_guide_length():
    with pytest.raises(ValueError):
        align_guide_to_candidates_gpu("A" * 19, [])


def test_invalid_candidate_length():
    candidates = [{"protospacer": "A" * 19}]

    with pytest.raises(ValueError):
        align_guide_to_candidates_gpu(GUIDE, candidates)


def test_missing_protospacer():
    with pytest.raises(ValueError):
        align_guide_to_candidates_gpu(
            GUIDE,
            [{"strand": "+"}],
        )


def test_invalid_candidate_collection():
    with pytest.raises(TypeError):
        align_guide_to_candidates_gpu(GUIDE, "invalid")


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        align_guide_to_candidates_gpu(
            GUIDE,
            [],
            batch_size=batch_size,
        )


def test_invalid_batch_size_type():
    with pytest.raises(TypeError):
        align_guide_to_candidates_gpu(
            GUIDE,
            [],
            batch_size=True,
        )


def test_invalid_mismatch_threshold():
    with pytest.raises(ValueError):
        align_guide_to_candidates_gpu(
            GUIDE,
            [],
            max_mismatches=-1,
        )


# -----------------------------------------
# CPU fallback tests
# -----------------------------------------

def test_cpu_fallback(monkeypatch):
    pytest.importorskip("numba")
    from numba import cuda

    # Simulate CUDA being unavailable.
    monkeypatch.setattr(
        cuda,
        "is_available",
        lambda: False,
    )

    candidates = [
        {"protospacer": GUIDE},
        {"protospacer": "C" + "A" * 19},
        {"protospacer": "CCCC" + "A" * 16},
    ]

    actual = align_guide_to_candidates_gpu(
        GUIDE,
        candidates,
        fallback_to_cpu=True,
    )

    expected = align_guide_to_candidates(
        GUIDE,
        candidates,
    )

    assert actual == expected


def test_cpu_fallback_disabled(monkeypatch):
    pytest.importorskip("numba")
    from numba import cuda

    monkeypatch.setattr(
        cuda,
        "is_available",
        lambda: False,
    )

    candidates = [{"protospacer": GUIDE}]

    with pytest.raises(RuntimeError):
        align_guide_to_candidates_gpu(
            GUIDE,
            candidates,
            fallback_to_cpu=False,
        )


# -----------------------------------------
# CUDA execution tests
# -----------------------------------------

def require_cuda():
    """Skip CUDA execution tests if unavailable."""
    pytest.importorskip("numba")
    from numba import cuda

    if not cuda.is_available():
        pytest.skip(
            "Requires CUDA hardware or simulation."
        )


def test_gpu_matches_cpu():
    require_cuda()

    candidates = [
        {"protospacer": "A" * 20},
        {"protospacer": "C" + "A" * 19},
        {"protospacer": "CCC" + "A" * 17},
        {"protospacer": "CCCC" + "A" * 16},
        {"protospacer": "N" + "A" * 19},
    ]

    gpu_results = align_guide_to_candidates_gpu(
        GUIDE,
        candidates,
        batch_size=2,
    )

    cpu_results = align_guide_to_candidates(
        GUIDE,
        candidates,
    )

    assert gpu_results == cpu_results


def test_gpu_batch_processing():
    require_cuda()

    candidates = [
        {
            "protospacer": "C" * (i % 5)
            + "A" * (20 - (i % 5)),
            "start": i * 23,
            "strand": "+" if i % 2 == 0 else "-",
        }
        for i in range(10)
    ]

    results = align_guide_to_candidates_gpu(
        GUIDE,
        candidates,
        batch_size=3,
    )

    expected = align_guide_to_candidates(
        GUIDE,
        candidates,
    )

    assert results == expected
    assert len(results) == 8


def test_gpu_preserves_metadata():
    require_cuda()

    candidate = {
        "chromosome_id": "chr1",
        "strand": "-",
        "start": 100,
        "end": 123,
        "protospacer_start": 103,
        "protospacer_end": 123,
        "pam_start": 100,
        "pam_end": 103,
        "protospacer": "C" + "A" * 19,
        "pam": "AGG",
    }

    results = align_guide_to_candidates_gpu(
        GUIDE,
        [candidate],
    )

    assert len(results) == 1
    assert results[0] == {
        **candidate,
        "mismatches": 1,
    }

    # Ensure the original dictionary is unchanged.
    assert "mismatches" not in candidate
