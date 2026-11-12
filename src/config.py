@'
"""Central configuration for GeneWeaver."""

PROJECT_NAME = "GeneWeaver"

# CRISPR configuration
DEFAULT_NUCLEASE = "SpCas9"
DEFAULT_PAM_PATTERN = "NGG"
DEFAULT_GUIDE_LENGTH = 20
DEFAULT_MAX_MISMATCHES = 3

# DNA sequence configuration
VALID_DNA_BASES = {"A", "T", "G", "C"}
AMBIGUOUS_DNA_BASES = {"N"}

# Genome processing configuration
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_CHUNK_OVERLAP = DEFAULT_GUIDE_LENGTH + len(DEFAULT_PAM_PATTERN)

# Output configuration
DEFAULT_OUTPUT_DIR = "results"
'@ | Set-Content src/config.py