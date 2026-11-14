
"""FASTA genome parsing utilities for GeneWeaver."""

from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path: str | Path) -> list[dict]:
    """
    Parse a FASTA file and extract sequence identifiers and DNA sequences.

    Args:
        file_path: Path to the input FASTA file.

    Returns:
        A list of dictionaries containing sequence IDs and sequences.

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        ValueError: If the file contains no valid FASTA records.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    records = []

    for record in SeqIO.parse(file_path, "fasta"):
        records.append({
            "id": record.id,
            "description": record.description,
            "sequence": str(record.seq).upper()
        })

    if not records:
        raise ValueError("No valid FASTA records found.")

    return records
