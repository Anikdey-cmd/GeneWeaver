from src.preprocessing.sequence_preprocessor import preprocess_sequence


def chunk_sequence(
    sequence: str,
    chromosome_id: str = "unknown",
    chunk_size: int = 100_000,
    overlap: int = 23
) -> list[dict]:

    if not isinstance(chromosome_id, str) or not chromosome_id.strip():
        raise ValueError("chromosome_id must be a non-empty string")

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    sequence = preprocess_sequence(sequence)

    step_size = chunk_size - overlap
    chunks = []

    for start in range(0, len(sequence), step_size):
        end = min(start + chunk_size, len(sequence))

        chunks.append({
            "chromosome_id": chromosome_id,
            "chunk_index": len(chunks),
            "start": start,
            "end": end,
            "sequence": sequence[start:end]
        })

        if end == len(sequence):
            break
    return chunks