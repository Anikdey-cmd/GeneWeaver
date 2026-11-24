
"""CPU-based CRISPR off-target detection pipeline for GeneWeaver."""

from pathlib import Path

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_GUIDE_LENGTH,
    DEFAULT_MAX_MISMATCHES,
    DEFAULT_PAM_PATTERN,
)

from src.parser.genomic_parser import parse_fasta
from src.preprocessing.sequence_preprocessor import preprocess_sequence
from src.chunking.genome_chunker import chunk_sequence
from src.matching.pam_scanner import find_pam_sites
from src.cpu.cpu_alignment import align_guide_to_candidates


# Coordinates returned by the PAM scanner are relative to each chunk.
COORDINATE_FIELDS = (
    "start",
    "end",
    "protospacer_start",
    "protospacer_end",
    "pam_start",
    "pam_end",
)


def run_cpu_pipeline(
    fasta_path: str | Path,
    guide: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> list[dict]:
    """
    Run the complete CPU-based CRISPR off-target detection pipeline.

    Steps:
        1. Validate inputs.
        2. Parse genomic FASTA records.
        3. Preprocess DNA sequences.
        4. Divide sequences into overlapping chunks.
        5. Detect PAM sites on both strands.
        6. Compare candidate protospacers against the guide.
        7. Convert coordinates to genome-relative positions.
        8. Remove duplicated candidate sites.

    Returns:
        A list of candidate off-target sites with chromosome ID,
        strand, genomic coordinates, PAM, protospacer,
        and mismatch count.

    Coordinates are zero-based and end-exclusive.
    """

    # Step 1: Normalize and validate the guide.
    guide = preprocess_sequence(guide)

    # Reuse existing alignment validation for the guide length
    # and mismatch threshold, even when no PAM sites are found.
    align_guide_to_candidates(
        guide,
        [],
        max_mismatches=max_mismatches,
    )

    # Step 2: Validate chunking parameters.
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")

    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    window_length = DEFAULT_GUIDE_LENGTH + len(DEFAULT_PAM_PATTERN)

    if chunk_size < window_length:
        raise ValueError(
            f"chunk_size must be at least {window_length}."
        )

    # Overlap must retain every possible complete target window.
    if overlap < window_length - 1:
        raise ValueError(
            f"overlap must be at least {window_length - 1} "
            "to avoid missing target sites at chunk boundaries."
        )

    # Step 3: Parse the FASTA file.
    records = parse_fasta(fasta_path)

    results = []
    seen_sites = set()

    # Step 4: Process each FASTA record independently.
    for record_index, record in enumerate(records):

        chromosome_id = record["id"]
        sequence = preprocess_sequence(record["sequence"])

        # Step 5: Divide the chromosome into overlapping chunks.
        chunks = chunk_sequence(
            sequence,
            chromosome_id=chromosome_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Step 6: Process every chunk.
        for chunk in chunks:

            # Detect potential SpCas9 sites on both DNA strands.
            candidates = find_pam_sites(
                chunk["sequence"],
                guide_length=DEFAULT_GUIDE_LENGTH,
                pam_pattern=DEFAULT_PAM_PATTERN,
            )

            # Compare the guide with candidate protospacers.
            alignments = align_guide_to_candidates(
                guide,
                candidates,
                max_mismatches=max_mismatches,
            )

            # Step 7: Convert local coordinates to genomic coordinates.
            for alignment in alignments:

                result = alignment.copy()

                for field in COORDINATE_FIELDS:
                    result[field] += chunk["start"]

                result["chromosome_id"] = chromosome_id

                # Step 8: Remove duplicates from overlapping chunks.
                # The record index distinguishes separate FASTA records,
                # even if two records have the same identifier.
                site_key = (
                    record_index,
                    result["strand"],
                    result["start"],
                    result["end"],
                )

                if site_key not in seen_sites:
                    seen_sites.add(site_key)
                    results.append(result)

    return results
