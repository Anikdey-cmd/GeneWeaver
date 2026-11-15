
"""DNA sequence preprocessing and validation utilities."""

DNA_COMPLEMENT = str.maketrans(
    "ATGCN",
    "TACGN"
)

VALID_DNA_BASES = set("ATGCN")


def normalize_sequence(sequence: str) -> str:
    """
    Remove whitespace and convert a DNA sequence to uppercase.

    Example:
        "atg c\n" -> "ATGC"
    """
    if not isinstance(sequence, str):
        raise TypeError("DNA sequence must be a string.")

    return "".join(sequence.split()).upper()


def validate_sequence(sequence: str) -> bool:
    """
    Validate a DNA sequence.

    Allowed bases:
        A = Adenine
        T = Thymine
        G = Guanine
        C = Cytosine
        N = Unknown or ambiguous nucleotide

    Returns True if valid, otherwise raises ValueError.
    """
    normalized = normalize_sequence(sequence)

    if not normalized:
        raise ValueError("DNA sequence cannot be empty.")

    invalid_bases = set(normalized) - VALID_DNA_BASES

    if invalid_bases:
        invalid = ", ".join(sorted(invalid_bases))
        raise ValueError(f"Invalid DNA characters found: {invalid}")

    return True


def preprocess_sequence(sequence: str) -> str:
    """
    Normalize and validate a DNA sequence.

    Returns the cleaned uppercase sequence.
    """
    normalized = normalize_sequence(sequence)
    validate_sequence(normalized)

    return normalized


def reverse_complement(sequence: str) -> str:
    """
    Generate the reverse complement of a DNA sequence.

    Example:
        ATGC -> GCAT
    """
    normalized = preprocess_sequence(sequence)

    return normalized.translate(DNA_COMPLEMENT)[::-1]
