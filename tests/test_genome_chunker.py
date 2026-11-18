import pytest

from src.chunking.genome_chunker import chunk_sequence


def test_basic_chunking():
    sequence = "A" * 250

    chunks = chunk_sequence(
        sequence,
        chromosome_id="chr1",
        chunk_size=100,
        overlap=20
    )

    assert len(chunks) == 3
    assert len(chunks[0]["sequence"]) == 100
    assert len(chunks[1]["sequence"]) == 100
    assert len(chunks[2]["sequence"]) == 90


def test_chunk_coordinates():
    sequence = "A" * 250

    chunks = chunk_sequence(sequence, chunk_size=100, overlap=20)

    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 100
    assert chunks[1]["start"] == 80
    assert chunks[1]["end"] == 180
    assert chunks[2]["start"] == 160
    assert chunks[2]["end"] == 250


def test_chromosome_id():
    chunks = chunk_sequence(
        "A" * 100,
        chromosome_id="chr5",
        chunk_size=50,
        overlap=10
    )

    assert all(chunk["chromosome_id"] == "chr5" for chunk in chunks)


def test_short_sequence():
    chunks = chunk_sequence("ATGC", chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["sequence"] == "ATGC"


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_sequence("ATGC", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_sequence("ATGC", chunk_size=-10)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_sequence("ATGC", chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        chunk_sequence("ATGC", chunk_size=10, overlap=-1)


def test_empty_sequence():
    with pytest.raises(ValueError):
        chunk_sequence("")


def test_chunk_content():
    sequence = "ATGC" * 50

    chunks = chunk_sequence(
        sequence,
        chunk_size=100,
        overlap=20
    )

    assert chunks[0]["sequence"] == sequence[0:100]
    assert chunks[1]["sequence"] == sequence[80:180]


def test_reconstruct_original_sequence():
    sequence = "ATGC" * 62 + "AT"

    chunks = chunk_sequence(
        sequence,
        chunk_size=100,
        overlap=20
    )

    reconstructed = chunks[0]["sequence"]

    for chunk in chunks[1:]:
        reconstructed += chunk["sequence"][20:]

    assert reconstructed == sequence