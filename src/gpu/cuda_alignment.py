
"""CUDA kernels for GeneWeaver DNA mismatch counting."""

from numba import cuda


# DNA encoding:
# A = 0, T = 1, G = 2, C = 3, N = 4

AMBIGUOUS_BASE_CODE = 4


@cuda.jit
def count_mismatches_kernel(
    guide_codes,
    candidate_codes,
    mismatch_counts
):
    """
    Count mismatches between a guide and genomic candidates.

    One CUDA thread processes one candidate.

    Expected inputs:
        guide_codes:
            1D integer array containing the encoded guide.

        candidate_codes:
            2D integer array with one candidate per row.

        mismatch_counts:
            1D output array containing mismatch counts.

    All sequences must have the same validated length.
    """

    candidate_index = cuda.grid(1)

    if candidate_index < candidate_codes.shape[0]:

        mismatches = 0

        for position in range(guide_codes.shape[0]):

            guide_base = guide_codes[position]
            target_base = candidate_codes[
                candidate_index, position
            ]

            if (
                guide_base == AMBIGUOUS_BASE_CODE
                or target_base == AMBIGUOUS_BASE_CODE
                or guide_base != target_base
            ):
                mismatches += 1

        mismatch_counts[candidate_index] = mismatches
