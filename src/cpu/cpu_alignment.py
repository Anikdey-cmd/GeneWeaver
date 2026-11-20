
"""CPU-based guide-to-genome alignment utilities."""
from src.config import DEFAULT_GUIDE_LENGTH, DEFAULT_MAX_MISMATCHES
from src.preprocessing.sequence_preprocessor import preprocess_sequence


def count_mismatches(guide: str, target: str) -> int:
    """
    Count nucleotide mismatches between two DNA sequences.

    Rules:
    - Both sequences must contain valid DNA bases.
    - Lowercase letters and whitespace are normalized.
    - Both sequences must have equal lengths.
    - Any position containing N counts as one mismatch,
      including N compared with N.

    Returns:
        Number of mismatched positions.
    """

    # Step 1: Normalize and validate both sequences.
    guide = preprocess_sequence(guide)
    target = preprocess_sequence(target)

    # Step 2: Check that both sequences have equal lengths.
    if len(guide) != len(target):
        raise ValueError("Guide and target must have equal lengths.")

    # Step 3: Compare corresponding nucleotides.
    mismatches = 0

    for guide_base, target_base in zip(guide, target):

        if guide_base == "N" or target_base == "N":
            mismatches += 1

        elif guide_base != target_base:
            mismatches += 1

    # Step 4: Return the mismatch count.
    return mismatches


def align_guide_to_candidates(
    guide: str,
    candidates: list[dict],
    max_mismatches: int = DEFAULT_MAX_MISMATCHES
) -> list[dict]:
    """
    Compare a guide with PAM-scanner candidate protospacers.

    Only candidates within the mismatch threshold are returned.
    Original candidate metadata and coordinates are preserved.
    """

    # Step 1: Validate and normalize the guide.
    guide = preprocess_sequence(guide)

    if len(guide) != DEFAULT_GUIDE_LENGTH:
        raise ValueError(
            f"Guide must contain exactly {DEFAULT_GUIDE_LENGTH} nucleotides."
        )

    # Step 2: Validate the mismatch threshold.
    if isinstance(max_mismatches, bool) or not isinstance(max_mismatches, int):
        raise TypeError("max_mismatches must be an integer.")

    if max_mismatches < 0:
        raise ValueError("max_mismatches cannot be negative.")

    # Step 3: Validate candidate collection.
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list.")

    results = []

    # Step 4: Compare the guide with each protospacer.
    for candidate in candidates:

        if not isinstance(candidate, dict):
            raise TypeError("Each candidate must be a dictionary.")

        if "protospacer" not in candidate:
            raise ValueError("Candidate is missing 'protospacer'.")

        mismatches = count_mismatches(
            guide,
            candidate["protospacer"]
        )

        # Step 5: Keep candidates within the threshold.
        if mismatches <= max_mismatches:

            # Copy to preserve original candidate information.
            alignment = candidate.copy()

            # Add the calculated mismatch count.
            alignment["mismatches"] = mismatches

            results.append(alignment)

    return results
