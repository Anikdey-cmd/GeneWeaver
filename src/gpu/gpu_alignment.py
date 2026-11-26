
"""GPU alignment wrapper with optional CPU fallback for GeneWeaver."""

import numpy as np

from src.config import (
    DEFAULT_GUIDE_LENGTH,
    DEFAULT_MAX_MISMATCHES,
)
from src.cpu.cpu_alignment import align_guide_to_candidates
from src.preprocessing.sequence_preprocessor import preprocess_sequence


DNA_CODES = {
    "A": 0,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 4,
}

THREADS_PER_BLOCK = 128
DEFAULT_BATCH_SIZE = 4096


def encode_dna(sequence: str) -> np.ndarray:
    """Validate DNA and encode nucleotides as 32-bit integers."""
    sequence = preprocess_sequence(sequence)

    return np.array(
        [DNA_CODES[base] for base in sequence],
        dtype=np.int32,
    )


def align_guide_to_candidates_gpu(
    guide: str,
    candidates: list[dict],
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fallback_to_cpu: bool = True,
) -> list[dict]:
    """
    Align a guide against candidate protospacers using CUDA.

    Uses the existing CPU implementation when CUDA is unavailable
    and fallback_to_cpu is True.

    Preserves candidate metadata, ordering and genomic coordinates.
    """

    # Step 1: Validate the guide.
    guide = preprocess_sequence(guide)

    if len(guide) != DEFAULT_GUIDE_LENGTH:
        raise ValueError(
            f"Guide must contain exactly {DEFAULT_GUIDE_LENGTH} nucleotides."
        )

    # Step 2: Validate processing parameters.
    if isinstance(max_mismatches, bool) or not isinstance(max_mismatches, int):
        raise TypeError("max_mismatches must be an integer.")

    if max_mismatches < 0:
        raise ValueError("max_mismatches cannot be negative.")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not isinstance(fallback_to_cpu, bool):
        raise TypeError("fallback_to_cpu must be a boolean.")

    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list.")

    # Step 3: Validate all candidate sequences.
    normalized_targets = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("Each candidate must be a dictionary.")

        if "protospacer" not in candidate:
            raise ValueError("Candidate is missing 'protospacer'.")

        target = preprocess_sequence(candidate["protospacer"])

        if len(target) != DEFAULT_GUIDE_LENGTH:
            raise ValueError(
                f"Candidate protospacer must contain "
                f"{DEFAULT_GUIDE_LENGTH} nucleotides."
            )

        normalized_targets.append(target)

    if not candidates:
        return []

    # Step 4: Check CUDA availability.
    # Import Numba lazily so CPU-only installations still work.
    try:
        from numba import cuda
        cuda_available = cuda.is_available()
    except (ImportError, OSError):
        cuda_available = False

    if not cuda_available:
        if fallback_to_cpu:
            return align_guide_to_candidates(
                guide,
                candidates,
                max_mismatches=max_mismatches,
            )

        raise RuntimeError(
            "CUDA is unavailable and CPU fallback is disabled."
        )

    # Import the kernel only after confirming CUDA availability.
    from src.gpu.cuda_alignment import count_mismatches_kernel

    # Step 5: Encode the guide and transfer it to CUDA.
    guide_codes = encode_dna(guide)
    device_guide = cuda.to_device(guide_codes)

    results = []

    # Step 6: Process candidate sequences in batches.
    for batch_start in range(0, len(candidates), batch_size):
        batch_end = min(
            batch_start + batch_size,
            len(candidates),
        )

        batch_targets = normalized_targets[batch_start:batch_end]

        candidate_codes = np.array(
            [
                [DNA_CODES[base] for base in target]
                for target in batch_targets
            ],
            dtype=np.int32,
        )

        # Transfer candidates to device memory.
        device_candidates = cuda.to_device(candidate_codes)

        # Allocate output memory on the CUDA device.
        batch_count = len(batch_targets)

        device_mismatches = cuda.device_array(
            batch_count,
            dtype=np.int32,
        )

        blocks_per_grid = (
            batch_count + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK

        # Step 7: Execute the CUDA mismatch-counting kernel.
        count_mismatches_kernel[
            blocks_per_grid,
            THREADS_PER_BLOCK
        ](
            device_guide,
            device_candidates,
            device_mismatches,
        )

        # Step 8: Copy mismatch counts back to CPU memory.
        mismatch_counts = device_mismatches.copy_to_host()

        # Step 9: Filter results and preserve candidate information.
        for index, mismatch_count in enumerate(mismatch_counts):
            if int(mismatch_count) <= max_mismatches:
                original_candidate = candidates[batch_start + index]

                alignment = original_candidate.copy()
                alignment["mismatches"] = int(mismatch_count)

                results.append(alignment)

    return results
