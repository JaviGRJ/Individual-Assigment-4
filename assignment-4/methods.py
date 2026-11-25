"""Reused Assignment 3 methods and the distributed Assignment 4 implementation."""

from __future__ import annotations

import os
import time
from typing import TypeAlias

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

Matrix: TypeAlias = list[list[float]]
_THREADPOOL_CONTROLLER = None


def zeros(n: int) -> Matrix:
    return [[0.0] * n for _ in range(n)]


def multiply_sequential(a: Matrix, b: Matrix) -> Matrix:
    """The validated i-j-k sequential baseline reused from Assignment 3."""
    n = len(a)
    c = zeros(n)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i][k] * b[k][j]
            c[i][j] = total
    return c


def _multiply_row(args: tuple[Matrix, Matrix, int]) -> list[float]:
    a, b, i = args
    n = len(a)
    row: list[float] = []
    for j in range(n):
        total = 0.0
        for k in range(n):
            total += a[i][k] * b[k][j]
        row.append(total)
    return row


def multiply_parallel(a: Matrix, b: Matrix, threads: int) -> Matrix:
    """The validated row-partitioned threaded version reused from Assignment 3."""
    from concurrent.futures import ThreadPoolExecutor

    if threads < 1:
        raise ValueError("threads must be at least one")
    n = len(a)
    arguments = ((a, b, i) for i in range(n))
    with ThreadPoolExecutor(max_workers=threads) as executor:
        return list(executor.map(_multiply_row, arguments))


def assert_close(
    actual: Matrix,
    expected: Matrix,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError("matrix dimensions differ")
    for i, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if len(actual_row) != len(expected_row):
            raise AssertionError("matrix dimensions differ")
        for j, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
            allowed = absolute_tolerance + relative_tolerance * abs(expected_value)
            if abs(actual_value - expected_value) > allowed:
                raise AssertionError(
                    f"mismatch at ({i}, {j}): {actual_value} != {expected_value}"
                )


def _limit_worker_blas_threads() -> dict[str, object]:
    global _THREADPOOL_CONTROLLER
    _THREADPOOL_CONTROLLER = threadpool_limits(limits=1)
    return {
        "pid": os.getpid(),
        "threadpools": [
            {
                "internal_api": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
                "prefix": pool.get("prefix"),
            }
            for pool in threadpool_info()
        ],
    }


def _worker_rss_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def _multiply_block(a_block: np.ndarray, b_block: np.ndarray) -> np.ndarray:
    return a_block @ b_block


def _sum_blocks(*blocks: np.ndarray) -> np.ndarray:
    result = np.zeros_like(blocks[0])
    for block in blocks:
        result += block
    return result


def partition_matrix(
    matrix: Matrix,
    block_size: int,
    prefix: str,
) -> tuple[dict[str, np.ndarray], int]:
    array = np.asarray(matrix, dtype=np.float64)
    n = array.shape[0]
    blocks: dict[str, np.ndarray] = {}
    block_count = (n + block_size - 1) // block_size
    for row in range(block_count):
        row_start = row * block_size
        row_end = min(row_start + block_size, n)
        for column in range(block_count):
            column_start = column * block_size
            column_end = min(column_start + block_size, n)
            blocks[f"{prefix}:{row}:{column}"] = array[
                row_start:row_end, column_start:column_end
            ].copy()
    return blocks, block_count


def _sample_worker_memory(client) -> list[int]:
    measurements = client.run(_worker_rss_bytes)
    return list(measurements.values())


def multiply_distributed(
    client,
    a: Matrix,
    b: Matrix,
    block_size: int,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Partition, scatter, multiply tiles on Dask workers, and collect C."""
    if block_size < 1:
        raise ValueError("block_size must be at least one")
    if len(a) != len(b) or any(len(row) != len(a) for row in a + b):
        raise ValueError("inputs must be equally sized square matrices")

    partition_started = time.perf_counter()
    a_blocks, block_count = partition_matrix(a, block_size, "A")
    b_blocks, _ = partition_matrix(b, block_size, "B")
    partition_ms = (time.perf_counter() - partition_started) * 1000.0
    input_bytes = sum(block.nbytes for block in a_blocks.values())
    input_bytes += sum(block.nbytes for block in b_blocks.values())

    scatter_started = time.perf_counter()
    a_futures = client.scatter(a_blocks, hash=False)
    b_futures = client.scatter(b_blocks, hash=False)
    input_scatter_ms = (time.perf_counter() - scatter_started) * 1000.0

    submission_started = time.perf_counter()
    products: dict[tuple[int, int, int], object] = {}
    for row in range(block_count):
        for column in range(block_count):
            for inner in range(block_count):
                products[(row, column, inner)] = client.submit(
                    _multiply_block,
                    a_futures[f"A:{row}:{inner}"],
                    b_futures[f"B:{inner}:{column}"],
                    pure=False,
                )

    outputs: dict[tuple[int, int], object] = {}
    for row in range(block_count):
        for column in range(block_count):
            terms = [
                products[(row, column, inner)]
                for inner in range(block_count)
            ]
            outputs[(row, column)] = client.submit(
                _sum_blocks,
                *terms,
                pure=False,
            )

    task_submission_ms = (time.perf_counter() - submission_started) * 1000.0
    wait_started = time.perf_counter()
    from distributed import wait

    _, not_done = wait(
        list(outputs.values()),
        timeout=timeout_seconds,
    )
    compute_wait_ms = (time.perf_counter() - wait_started) * 1000.0
    if not_done:
        client.cancel(list(not_done), force=True)
        return {
            "result": None,
            "status": "timeout",
            "input_payload_bytes": input_bytes,
            "result_payload_bytes": 0,
            "block_count_per_dimension": block_count,
            "task_count": len(products) + len(outputs),
            "partition_ms": partition_ms,
            "input_scatter_ms": input_scatter_ms,
            "task_submission_ms": task_submission_ms,
            "compute_wait_ms": compute_wait_ms,
            "result_collection_ms": 0.0,
        }

    gather_started = time.perf_counter()
    completed = client.gather(list(outputs.values()))
    result_collection_ms = (time.perf_counter() - gather_started) * 1000.0
    n = len(a)
    result = np.zeros((n, n), dtype=np.float64)
    for ((row, column), block) in zip(outputs, completed):
        row_start = row * block_size
        column_start = column * block_size
        result[
            row_start:row_start + block.shape[0],
            column_start:column_start + block.shape[1],
        ] = block

    return {
        "result": result.tolist(),
        "input_payload_bytes": input_bytes,
        "result_payload_bytes": int(result.nbytes),
        "block_count_per_dimension": block_count,
        "task_count": len(products) + len(outputs),
        "status": "ok",
        "partition_ms": partition_ms,
        "input_scatter_ms": input_scatter_ms,
        "task_submission_ms": task_submission_ms,
        "compute_wait_ms": compute_wait_ms,
        "result_collection_ms": result_collection_ms,
    }


def self_test() -> None:
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert_close(multiply_sequential(a, b), expected)
    for threads in (1, 2):
        assert_close(multiply_parallel(a, b, threads), expected)


if __name__ == "__main__":
    self_test()
    print("Assignment 4 sequential/parallel reference tests passed.")
