
from src.preprocessing.sequence_preprocessor import preprocess_sequence


def find_pam_sites(
    sequence: str,
    guide_length: int = 20,
    pam_pattern: str = "NGG"
) -> list[dict]:
    """
    Find candidate SpCas9 target sites on both DNA strands.

    Coordinates are zero-based and end-exclusive.
    """

    sequence = preprocess_sequence(sequence)

    if isinstance(guide_length, bool) or not isinstance(guide_length, int):
        raise TypeError("guide_length must be an integer")

    if guide_length <= 0:
        raise ValueError("guide_length must be positive")

    if not isinstance(pam_pattern, str):
        raise TypeError("pam_pattern must be a string")

    pam_pattern = pam_pattern.upper()

    if not pam_pattern or any(base not in "ATGCN" for base in pam_pattern):
        raise ValueError("Invalid PAM pattern")

    pam_length = len(pam_pattern)
    window_length = guide_length + pam_length
    results = []

    # Forward strand: protospacer followed by PAM.
    for start in range(len(sequence) - window_length + 1):
        guide_start = start
        pam_start = start + guide_length
        pam_end = pam_start + pam_length

        protospacer = sequence[guide_start:pam_start]
        pam = sequence[pam_start:pam_end]

        if all(
            pattern_base == "N" or pattern_base == actual_base
            for pattern_base, actual_base in zip(pam_pattern, pam)
        ):
            results.append({
                "strand": "+",
                "start": guide_start,
                "end": pam_end,
                "protospacer_start": guide_start,
                "protospacer_end": pam_start,
                "pam_start": pam_start,
                "pam_end": pam_end,
                "protospacer": protospacer,
                "pam": pam
            })

    # Reverse strand:
    # The reverse-complemented genomic window must end in the PAM.
    for start in range(len(sequence) - window_length + 1):
        end = start + window_length
        genomic_window = sequence[start:end]

        reverse_window = genomic_window.translate(
            str.maketrans("ATGCN", "TACGN")
        )[::-1]

        protospacer = reverse_window[:guide_length]
        pam = reverse_window[guide_length:]

        if all(
            pattern_base == "N" or pattern_base == actual_base
            for pattern_base, actual_base in zip(pam_pattern, pam)
        ):
            results.append({
                "strand": "-",
                "start": start,
                "end": end,
                "protospacer_start": start + pam_length,
                "protospacer_end": end,
                "pam_start": start,
                "pam_end": start + pam_length,
                "protospacer": protospacer,
                "pam": pam
            })

    return sorted(results, key=lambda site: (site["start"], site["strand"]))
