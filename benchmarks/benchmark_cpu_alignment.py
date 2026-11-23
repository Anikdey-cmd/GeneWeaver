
"""Benchmark CPU-based CRISPR guide alignment performance."""

import platform
import sys
import time
from statistics import median

from src.config import DEFAULT_GUIDE_LENGTH, DEFAULT_MAX_MISMATCHES
from src.cpu.cpu_alignment import align_guide_to_candidates


GUIDE = "A" * DEFAULT_GUIDE_LENGTH

CANDIDATE_COUNTS = [1_000, 10_000, 100_000]
REPEAT_COUNT = 5


def generate_candidates(count: int) -> list[dict]:
    """Generate reproducible synthetic DNA candidates."""

    candidates = []

    for index in range(count):
        mismatch_count = index % 5

        protospacer = (
            "C" * mismatch_count
            + "A" * (DEFAULT_GUIDE_LENGTH - mismatch_count)
        )

        candidates.append({
            "protospacer": protospacer,
            "strand": "+" if index % 2 == 0 else "-",
            "start": index * 23,
            "end": index * 23 + 23,
            "pam": "AGG"
        })

    return candidates


def run_benchmark() -> None:
    """Measure CPU alignment execution time and throughput."""

    print("\nGeneWeaver CPU Alignment Benchmark")
    print("-" * 60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Guide length: {DEFAULT_GUIDE_LENGTH}")
    print(f"Maximum mismatches: {DEFAULT_MAX_MISMATCHES}")
    print(f"Repeats per experiment: {REPEAT_COUNT}")

    # Warm up the function before measuring.
    warmup_candidates = generate_candidates(100)
    align_guide_to_candidates(GUIDE, warmup_candidates)

    for count in CANDIDATE_COUNTS:
        # Candidate generation is not included in the timing.
        candidates = generate_candidates(count)

        expected_matches = sum(
            index % 5 <= DEFAULT_MAX_MISMATCHES
            for index in range(count)
        )

        execution_times = []

        for _ in range(REPEAT_COUNT):
            start_time = time.perf_counter()

            results = align_guide_to_candidates(
                GUIDE,
                candidates,
                max_mismatches=DEFAULT_MAX_MISMATCHES
            )

            elapsed_time = time.perf_counter() - start_time

            if len(results) != expected_matches:
                raise AssertionError(
                    "Unexpected alignment result count."
                )

            execution_times.append(elapsed_time)

        median_time = median(execution_times)
        throughput = count / median_time

        print(f"\nCandidates: {count:,}")
        print(f"Matching candidates: {expected_matches:,}")
        print(f"Median execution time: {median_time:.6f} seconds")
        print(f"Minimum execution time: {min(execution_times):.6f} seconds")
        print(f"Maximum execution time: {max(execution_times):.6f} seconds")
        print(f"Throughput: {throughput:,.2f} candidates/second")

    print("\nCPU benchmarking completed.")


if __name__ == "__main__":
    run_benchmark()
