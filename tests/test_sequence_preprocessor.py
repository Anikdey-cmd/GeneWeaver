
import pytest

from src.preprocessing.sequence_preprocessor import (
    normalize_sequence,
    validate_sequence,
    preprocess_sequence,
    reverse_complement,
)


def test_normalize_sequence():
    assert normalize_sequence(" atg c\n") == "ATGC"


def test_validate_valid_sequence():
    assert validate_sequence("ATGCN") is True


def test_validate_lowercase_sequence():
    assert validate_sequence("atgcn") is True


def test_invalid_sequence():
    with pytest.raises(ValueError):
        validate_sequence("ATGX")


def test_empty_sequence():
    with pytest.raises(ValueError):
        validate_sequence("   ")


def test_preprocess_sequence():
    assert preprocess_sequence(" atg c n ") == "ATGCN"


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"


def test_reverse_complement_with_ambiguous_base():
    assert reverse_complement("ATGN") == "NCAT"


def test_non_string_input():
    with pytest.raises(TypeError):
        normalize_sequence(123)
