
"""Unit tests for the GeneWeaver CPU pipeline."""

import pytest

from src.cpu.cpu_pipeline import run_cpu_pipeline
from src.preprocessing.sequence_preprocessor import reverse_complement


GUIDE = "A" * 20


def create_fasta(tmp_path, content):
    """Create a temporary FASTA file for testing."""
    fasta_path = tmp_path / "test_genome.fasta"
    fasta_path.write_text(content, encoding="utf-8")
    return fasta_path


def test_forward_strand_pipeline(tmp_path):
    sequence = "A" * 10 + GUIDE + "AGG" + "A" * 10
    fasta = create_fasta(tmp_path, f">chr1\n{sequence}\n")

    results = run_cpu_pipeline(fasta, GUIDE)

    assert len(results) == 1

    site = results[0]

    assert site["chromosome_id"] == "chr1"
    assert site["strand"] == "+"
    assert site["start"] == 10
    assert site["end"] == 33
    assert site["protospacer"] == GUIDE
    assert site["pam"] == "AGG"
    assert site["mismatches"] == 0


def test_reverse_strand_pipeline(tmp_path):
    reverse_site = reverse_complement(GUIDE + "AGG")
    sequence = "A" * 10 + reverse_site + "A" * 10

    fasta = create_fasta(tmp_path, f">chr1\n{sequence}\n")

    results = run_cpu_pipeline(fasta, GUIDE)

    assert len(results) == 1

    site = results[0]

    assert site["strand"] == "-"
    assert site["start"] == 10
    assert site["end"] == 33
    assert site["pam_start"] == 10
    assert site["pam_end"] == 13
    assert site["protospacer_start"] == 13
    assert site["protospacer_end"] == 33
    assert site["mismatches"] == 0


def test_both_strands_pipeline(tmp_path):
    forward_site = GUIDE + "AGG"
    reverse_site = reverse_complement(GUIDE + "AGG")

    sequence = (
        "A" * 4
        + forward_site
        + "A" * 10
        + reverse_site
        + "A" * 5
    )

    fasta = create_fasta(tmp_path, f">chr1\n{sequence}\n")

    results = run_cpu_pipeline(fasta, GUIDE)

    assert len(results) == 2
    assert {site["strand"] for site in results} == {"+", "-"}
    assert all(site["mismatches"] == 0 for site in results)


def test_multiple_chromosomes(tmp_path):
    chr1 = "A" * 5 + GUIDE + "AGG"
    chr2 = "A" * 7 + GUIDE + "TGG"

    fasta = create_fasta(
        tmp_path,
        f">chr1\n{chr1}\n>chr2\n{chr2}\n"
    )

    results = run_cpu_pipeline(fasta, GUIDE)

    assert len(results) == 2

    by_chromosome = {
        site["chromosome_id"]: site for site in results
    }

    assert by_chromosome["chr1"]["start"] == 5
    assert by_chromosome["chr2"]["start"] == 7
    assert by_chromosome["chr1"]["pam"] == "AGG"
    assert by_chromosome["chr2"]["pam"] == "TGG"


def test_duplicate_removal(tmp_path):
    # The site is fully contained in two overlapping chunks.
    sequence = "A" * 35 + GUIDE + "AGG" + "A" * 32
    fasta = create_fasta(tmp_path, f">chr1\n{sequence}\n")

    results = run_cpu_pipeline(
        fasta,
        GUIDE,
        chunk_size=60,
        overlap=30,
    )

    assert len(results) == 1
    assert results[0]["start"] == 35
    assert results[0]["end"] == 58


def test_site_crossing_chunk_boundary(tmp_path):
    # A target at position 24 crosses a chunk boundary
    # but must still be detected because of overlap.
    sequence = "A" * 24 + GUIDE + "AGG" + "A" * 20
    fasta = create_fasta(tmp_path, f">chr1\n{sequence}\n")

    results = run_cpu_pipeline(
        fasta,
        GUIDE,
        chunk_size=30,
        overlap=22,
    )

    assert len(results) == 1
    assert results[0]["start"] == 24
    assert results[0]["end"] == 47


def test_mismatch_filtering(tmp_path):
    target = "C" + "A" * 19
    fasta = create_fasta(
        tmp_path,
        f">chr1\n{target}AGG\n"
    )

    accepted = run_cpu_pipeline(
        fasta,
        GUIDE,
        max_mismatches=1,
    )

    rejected = run_cpu_pipeline(
        fasta,
        GUIDE,
        max_mismatches=0,
    )

    assert len(accepted) == 1
    assert accepted[0]["mismatches"] == 1
    assert rejected == []


def test_no_pam_sites(tmp_path):
    fasta = create_fasta(tmp_path, f">chr1\n{'A' * 60}\n")

    results = run_cpu_pipeline(fasta, GUIDE)

    assert results == []


def test_duplicate_chromosome_ids(tmp_path):
    sequence = GUIDE + "AGG"

    fasta = create_fasta(
        tmp_path,
        f">chr1\n{sequence}\n>chr1\n{sequence}\n"
    )

    results = run_cpu_pipeline(fasta, GUIDE)

    # Two separate FASTA records must remain distinguishable.
    assert len(results) == 2


def test_invalid_guide_length(tmp_path):
    fasta = create_fasta(tmp_path, f">chr1\n{GUIDE}AGG\n")

    with pytest.raises(ValueError):
        run_cpu_pipeline(fasta, "A" * 19)


def test_invalid_chunk_size(tmp_path):
    fasta = create_fasta(tmp_path, f">chr1\n{GUIDE}AGG\n")

    with pytest.raises(ValueError):
        run_cpu_pipeline(
            fasta,
            GUIDE,
            chunk_size=22,
        )


def test_insufficient_overlap(tmp_path):
    fasta = create_fasta(tmp_path, f">chr1\n{GUIDE}AGG\n")

    with pytest.raises(ValueError):
        run_cpu_pipeline(
            fasta,
            GUIDE,
            chunk_size=30,
            overlap=21,
        )


def test_missing_fasta_file(tmp_path):
    missing_file = tmp_path / "missing.fasta"

    with pytest.raises(FileNotFoundError):
        run_cpu_pipeline(missing_file, GUIDE)
