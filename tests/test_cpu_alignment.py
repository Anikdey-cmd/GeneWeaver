
"""Unit tests for CPU-based DNA mismatch counting."""

import pytest
from src.matching.pam_scanner import find_pam_sites
from src.preprocessing.sequence_preprocessor import reverse_complement
from src.cpu.cpu_alignment import align_guide_to_candidates

from src.cpu.cpu_alignment import count_mismatches


def test_identical_sequences():
    guide = "ATGC" * 5
    target = "ATGC" * 5

    assert count_mismatches(guide, target) == 0


def test_single_mismatch():
    guide = "A" * 20
    target = "C" + "A" * 19

    assert count_mismatches(guide, target) == 1


def test_three_mismatches():
    guide = "A" * 20
    target = "CCC" + "A" * 17

    assert count_mismatches(guide, target) == 3


def test_all_positions_mismatched():
    guide = "A" * 20
    target = "T" * 20

    assert count_mismatches(guide, target) == 20


def test_lowercase_and_whitespace():
    guide = "atgc atgc"
    target = "ATGCATGC"

    assert count_mismatches(guide, target) == 0


def test_ambiguous_nucleotide():
    guide = "A" * 20
    target = "N" + "A" * 19

    assert count_mismatches(guide, target) == 1


def test_ambiguous_n_compared_with_n():
    guide = "N" + "A" * 19
    target = "N" + "A" * 19

    assert count_mismatches(guide, target) == 1


def test_unequal_sequence_lengths():
    guide = "A" * 20
    target = "A" * 19

    with pytest.raises(ValueError):
        count_mismatches(guide, target)


def test_invalid_dna_character():
    guide = "A" * 19 + "X"
    target = "A" * 20

    with pytest.raises(ValueError):
        count_mismatches(guide, target)


def test_invalid_input_type():
    with pytest.raises(TypeError):
        count_mismatches(123, "A" * 20)


def test_empty_sequence():
    with pytest.raises(ValueError):
        count_mismatches("", "A" * 20)



# --------------------------------------------------
# Tests for align_guide_to_candidates()
# --------------------------------------------------


def test_alignment_accepts_zero_and_three_mismatches():
    guide = "A" * 20

    candidates = [
        {"protospacer": "A" * 20, "strand": "+"},
        {"protospacer": "CCC" + "A" * 17, "strand": "-"}
    ]

    results = align_guide_to_candidates(guide, candidates)

    assert len(results) == 2
    assert [site["mismatches"] for site in results] == [0, 3]


def test_alignment_filters_more_than_three_mismatches():
    guide = "A" * 20

    candidates = [
        {"protospacer": "CCCC" + "A" * 16}
    ]

    results = align_guide_to_candidates(guide, candidates)

    assert results == []


def test_alignment_custom_mismatch_threshold():
    guide = "A" * 20

    candidates = [
        {"protospacer": "C" + "A" * 19},
        {"protospacer": "CC" + "A" * 18}
    ]

    results = align_guide_to_candidates(
        guide, candidates, max_mismatches=1
    )

    assert len(results) == 1
    assert results[0]["mismatches"] == 1


def test_alignment_preserves_metadata():
    guide = "A" * 20

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
        "pam": "AGG"
    }

    results = align_guide_to_candidates(guide, [candidate])

    assert len(results) == 1
    assert results[0] == {**candidate, "mismatches": 1}

    # Original candidate must not be modified.
    assert "mismatches" not in candidate
    assert results[0] is not candidate


def test_alignment_with_forward_pam_scanner():
    guide = "ATGC" * 5
    sequence = guide + "AGG"

    candidates = find_pam_sites(sequence)
    results = align_guide_to_candidates(guide, candidates)

    assert len(results) == 1
    assert results[0]["strand"] == "+"
    assert results[0]["mismatches"] == 0
    assert results[0]["start"] == 0
    assert results[0]["end"] == 23


def test_alignment_with_reverse_pam_scanner():
    guide = "ATGC" * 5

    # Create a genomic window containing a reverse-strand site.
    sequence = reverse_complement(guide + "AGG")

    candidates = find_pam_sites(sequence)
    results = align_guide_to_candidates(guide, candidates)

    assert len(results) == 1
    assert results[0]["strand"] == "-"
    assert results[0]["mismatches"] == 0
    assert results[0]["protospacer_start"] == 3
    assert results[0]["protospacer_end"] == 23


def test_alignment_empty_candidates():
    results = align_guide_to_candidates("A" * 20, [])

    assert results == []


def test_alignment_invalid_guide_length():
    with pytest.raises(ValueError):
        align_guide_to_candidates("A" * 19, [])


@pytest.mark.parametrize(
    "threshold, expected_error",
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError)
    ]
)
def test_alignment_invalid_threshold(threshold, expected_error):
    with pytest.raises(expected_error):
        align_guide_to_candidates(
            "A" * 20, [], max_mismatches=threshold
        )


def test_alignment_invalid_candidate_collection():
    with pytest.raises(TypeError):
        align_guide_to_candidates("A" * 20, "not a list")


def test_alignment_missing_protospacer():
    with pytest.raises(ValueError):
        align_guide_to_candidates("A" * 20, [{"strand": "+"}])


def test_alignment_invalid_protospacer_length():
    with pytest.raises(ValueError):
        align_guide_to_candidates(
            "A" * 20,
            [{"protospacer": "A" * 19}]
        )
