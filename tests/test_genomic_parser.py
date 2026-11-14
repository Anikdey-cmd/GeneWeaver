
from src.parser.genomic_parser import parse_fasta


def test_parse_fasta():
    records = parse_fasta("data/sample/sample_genome.fasta")

    assert len(records) == 3
    assert records[0]["id"] == "chr1"
    assert records[0]["sequence"] == "ATGCGTACGTAGCTAGCTAG"
    assert records[1]["id"] == "chr2"


def test_sequences_are_uppercase():
    records = parse_fasta("data/sample/sample_genome.fasta")

    for record in records:
        assert record["sequence"] == record["sequence"].upper()
