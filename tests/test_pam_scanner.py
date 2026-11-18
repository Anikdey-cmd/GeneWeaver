
import pytest

from src.matching.pam_scanner import find_pam_sites


def test_forward_strand_pam_detection():
    sequence = "A" * 10 + "ATGCATGCATGCATGCATGC" + "AGG"

    sites = find_pam_sites(sequence)

    assert len(sites) == 1
    assert sites[0]["strand"] == "+"
    assert sites[0]["protospacer"] == "ATGCATGCATGCATGCATGC"
    assert sites[0]["pam"] == "AGG"


def test_forward_strand_coordinates():
    sequence = "A" * 10 + "ATGCATGCATGCATGCATGC" + "AGG"

    sites = find_pam_sites(sequence)

    assert sites[0]["start"] == 10
    assert sites[0]["end"] == 33
    assert sites[0]["pam_start"] == 30
    assert sites[0]["pam_end"] == 33


def test_reverse_strand_pam_detection():
    # Reverse complement of a 20-nt protospacer followed by AGG.
    reverse_window = "ATGCATGCATGCATGCATGCAGG"
    genomic_window = reverse_window.translate(
        str.maketrans("ATGCN", "TACGN")
    )[::-1]

    sites = find_pam_sites(genomic_window)

    reverse_sites = [site for site in sites if site["strand"] == "-"]

    assert len(reverse_sites) == 1
    assert reverse_sites[0]["pam"] == "AGG"
    assert reverse_sites[0]["protospacer"] == "ATGCATGCATGCATGCATGC"


def test_reverse_strand_coordinates():
    reverse_window = "ATGCATGCATGCATGCATGCAGG"
    genomic_window = reverse_window.translate(
        str.maketrans("ATGCN", "TACGN")
    )[::-1]

    sites = find_pam_sites(genomic_window)

    reverse_site = next(site for site in sites if site["strand"] == "-")

    assert reverse_site["start"] == 0
    assert reverse_site["end"] == 23
    assert reverse_site["pam_start"] == 0
    assert reverse_site["pam_end"] == 3
    assert reverse_site["protospacer_start"] == 3
    assert reverse_site["protospacer_end"] == 23


def test_no_pam_found():
    sequence = "A" * 100

    sites = find_pam_sites(sequence)

    assert sites == []


def test_multiple_pam_sites():
    first = "ATGCATGCATGCATGCATGCAGG"
    second = "TGCATGCATGCATGCATGCATAAGG"

    sites = find_pam_sites(first + "AAAA" + second)

    assert len(sites) >= 2


def test_invalid_guide_length():
    with pytest.raises(ValueError):
        find_pam_sites("ATGC", guide_length=0)

    with pytest.raises(TypeError):
        find_pam_sites("ATGC", guide_length=2.5)


def test_invalid_pam_pattern():
    with pytest.raises(ValueError):
        find_pam_sites("ATGC", pam_pattern="XYZ")

    with pytest.raises(TypeError):
        find_pam_sites("ATGC", pam_pattern=123)
