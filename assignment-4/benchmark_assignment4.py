"""Reproducible benchmark for Assignment 4 distributed multiplication."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import psutil
from distributed import Client, LocalCluster

from methods import (
    _limit_worker_blas_threads,
    _worker_rss_bytes,
    assert_close,
    multiply_distributed,
    multiply_parallel,
    multiply_sequential,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SIZES = [32, 64, 128, 256]
DEFAULT_WORKERS = [1]
THREADS_PER_WORKER = 1
BLOCK_SIZE = 32
SEED = 20261124
REPETITIONS = 5
WARMUPS = 1
TIME_LIMIT_SECONDS = 60.0
MEMORY_LIMIT_MIB = 2048.0

FIELDS = [
    "size",
    "method",
    "implementation",
    "workers",
    "worker_processes",
    "physical_machines",
    "threads_per_worker",
    "repetition",
    "end_to_end_ms",
    "setup_ms",
    "partition_ms",
    "input_scatter_ms",
    "task_submission_ms",
    "compute_wait_ms",
    "result_collection_ms",
    "monitoring_ms",
    "shutdown_ms",
    "input_payload_mib",
    "result_payload_mib",
    "worker_rss_max_sampled_mib",
    "worker_rss_total_sampled_mib",
    "memory_value_mib",
    "memory_metric",
    "task_count",
    "status",
    "speedup_vs_sequential",
    "strong_speedup_vs_one_worker",
    "parallel_efficiency",
]


def make_matrix(rng: random.Random, n: int) -> list[list[float]]:
    return [[float(rng.randint(-8, 8)) for _ in range(n)] for _ in range(n)]


def calculate_iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def physical_core_count() -> int | None:
    if os.name != "nt":
        return psutil.cpu_count(logical=False)
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object -Property NumberOfCores -Sum).Sum",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def cpu_model_name() -> str:
    if os.name != "nt":
        return platform.processor() or "Not reported by the operating system"
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_Processor | "
            "Select-Object -First 1 -ExpandProperty Name)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def benchmark_local(method, arguments: tuple, expected) -> dict[str, object]:
    for _ in range(WARMUPS):
        assert_close(method(*arguments), expected)

    samples: list[float] = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        result = method(*arguments)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert_close(result, expected)
        samples.append(elapsed_ms)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        memory_result = method(*arguments)
        assert_close(memory_result, expected)
        memory_mib = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    finally:
        tracemalloc.stop()

    return {
        "samples": samples,
        "memory_value_mib": memory_mib,
        "memory_metric": "peak traced Python allocations (tracemalloc)",
        "status": "memory_limit" if memory_mib > MEMORY_LIMIT_MIB else "ok",
    }


def run_distributed_once(
    a: list[list[float]],
    b: list[list[float]],
    expected: list[list[float]],
    workers: int,
) -> dict[str, object]:
    started_all = time.perf_counter()
    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=THREADS_PER_WORKER,
        processes=True,
        dashboard_address=None,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    try:
        worker_count = len(client.scheduler_info()["workers"])
        worker_threadpools = client.run(_limit_worker_blas_threads)
        setup_ms = (time.perf_counter() - started_all) * 1000.0
        distributed_result = multiply_distributed(
            client,
            a,
            b,
            BLOCK_SIZE,
            TIME_LIMIT_SECONDS,
        )
        monitor_started = time.perf_counter()
        worker_memory_values = list(client.run(_worker_rss_bytes).values())
        if len(worker_memory_values) != worker_count:
            raise RuntimeError(
                "Could not sample RSS from every Dask worker process."
            )
        monitoring_ms = (time.perf_counter() - monitor_started) * 1000.0
    finally:
        shutdown_started = time.perf_counter()
        try:
            client.close()
        finally:
            cluster.close()
        shutdown_ms = (time.perf_counter() - shutdown_started) * 1000.0

    end_to_end_ms = (time.perf_counter() - started_all) * 1000.0
    end_to_end_ms -= monitoring_ms
    if distributed_result["status"] == "ok":
        assert_close(distributed_result["result"], expected)

    worker_memory_peak_mib = max(worker_memory_values) / (1024 * 1024)
    worker_memory_total_peak_mib = sum(worker_memory_values) / (1024 * 1024)
    status = str(distributed_result["status"])
    if status == "ok" and worker_memory_total_peak_mib > MEMORY_LIMIT_MIB:
        status = "memory_limit"
    return {
        "status": status,
        "end_to_end_ms": end_to_end_ms,
        "setup_ms": setup_ms,
        "partition_ms": distributed_result["partition_ms"],
        "input_scatter_ms": distributed_result["input_scatter_ms"],
        "task_submission_ms": distributed_result["task_submission_ms"],
        "compute_wait_ms": distributed_result["compute_wait_ms"],
        "result_collection_ms": distributed_result["result_collection_ms"],
        "monitoring_ms": monitoring_ms,
        "shutdown_ms": shutdown_ms,
        "worker_rss_max_sampled_mib": worker_memory_peak_mib,
        "worker_rss_total_sampled_mib": worker_memory_total_peak_mib,
        "input_payload_mib": distributed_result["input_payload_bytes"] / (1024 * 1024),
        "result_payload_mib": distributed_result["result_payload_bytes"] / (1024 * 1024),
        "worker_count": worker_count,
        "task_count": distributed_result["task_count"],
        "worker_threadpools": worker_threadpools,
    }


def run_benchmark(sizes: list[int], worker_counts: list[int]) -> dict[str, int]:
    rng = random.Random(SEED)
    pending: list[dict[str, object]] = []
    distributed_configs: list[dict[str, object]] = []

    for n in sizes:
        a = make_matrix(rng, n)
        b = make_matrix(rng, n)
        expected = multiply_sequential(a, b)

        for method_name, method, arguments in (
            ("sequential", multiply_sequential, (a, b)),
            ("threaded", multiply_parallel, (a, b, 4)),
        ):
            result = benchmark_local(method, arguments, expected)
            for repetition, time_ms in enumerate(result["samples"], 1):
                pending.append(
                    {
                        "size": n,
                        "method": method_name,
                        "implementation": (
                            "assignment-3 i-j-k baseline"
                            if method_name == "sequential"
                            else "assignment-3 ThreadPoolExecutor, 4 threads"
                        ),
                        "workers": "",
                        "worker_processes": "",
                        "physical_machines": 1,
                        "threads_per_worker": 4 if method_name == "threaded" else 1,
                        "repetition": repetition,
                        "end_to_end_ms": time_ms,
                        "status": result["status"],
                        "memory_value_mib": result["memory_value_mib"],
                        "memory_metric": result["memory_metric"],
                        "worker_rss_max_sampled_mib": "",
                        "worker_rss_total_sampled_mib": "",
                    }
                )

        for workers in worker_counts:
            warmup = run_distributed_once(a, b, expected, workers)
            if warmup["status"] != "ok":
                raise RuntimeError(
                    f"Distributed warm-up failed for n={n}, workers={workers}: "
                    f"{warmup['status']}"
                )
            samples: list[dict[str, object]] = []
            for _ in range(REPETITIONS):
                sample = run_distributed_once(a, b, expected, workers)
                samples.append(sample)
                if sample["status"] not in ("ok", "timeout"):
                    raise RuntimeError(
                        f"Distributed run failed for n={n}, workers={workers}: "
                        f"{sample['status']}"
                    )

            for repetition, sample in enumerate(samples, 1):
                pending.append(
                    {
                        "size": n,
                        "method": "distributed",
                        "implementation": (
                            "Dask Distributed LocalCluster; NumPy float64 tile "
                            "multiplication; one thread per worker"
                        ),
                        "workers": workers,
                        "worker_processes": sample["worker_count"],
                        "physical_machines": 1,
                        "threads_per_worker": THREADS_PER_WORKER,
                        "repetition": repetition,
                        "end_to_end_ms": sample["end_to_end_ms"],
                        "setup_ms": sample["setup_ms"],
                        "partition_ms": sample["partition_ms"],
                        "input_scatter_ms": sample["input_scatter_ms"],
                        "task_submission_ms": sample["task_submission_ms"],
                        "compute_wait_ms": sample["compute_wait_ms"],
                        "result_collection_ms": sample["result_collection_ms"],
                        "monitoring_ms": sample["monitoring_ms"],
                        "shutdown_ms": sample["shutdown_ms"],
                        "input_payload_mib": sample["input_payload_mib"],
                        "result_payload_mib": sample["result_payload_mib"],
                        "worker_rss_max_sampled_mib": sample[
                            "worker_rss_max_sampled_mib"
                        ],
                        "worker_rss_total_sampled_mib": sample[
                            "worker_rss_total_sampled_mib"
                        ],
                        "memory_metric": (
                            "sampled worker-process RSS; max and sum across worker "
                            "processes, not a continuous peak"
                        ),
                        "task_count": sample["task_count"],
                        "status": sample["status"],
                    }
                )
            distributed_configs.append(
                {
                    "size": n,
                    "workers": workers,
                    "processes": workers,
                    "threads_per_worker": THREADS_PER_WORKER,
                    "worker_threadpools": samples[-1]["worker_threadpools"],
                }
            )

    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = {}
    for row in pending:
        worker_key = int(row["workers"]) if row["workers"] != "" else 0
        grouped.setdefault(
            (int(row["size"]), str(row["method"]), worker_key), []
        ).append(row)

    sequential_medians: dict[int, float] = {}
    distributed_medians: dict[tuple[int, int], float] = {}
    method_medians: dict[tuple[int, str, int], float] = {}
    for key, rows in grouped.items():
        good = [
            float(row["end_to_end_ms"])
            for row in rows
            if row["status"] == "ok" and row.get("end_to_end_ms") not in (None, "")
        ]
        if not good:
            continue
        median = statistics.median(good)
        method_medians[key] = median
        if key[1] == "sequential":
            sequential_medians[key[0]] = median
        elif key[1] == "distributed":
            distributed_medians[(key[0], key[2])] = median

    for row in pending:
        size = int(row["size"])
        baseline = sequential_medians.get(size)
        worker_key = int(row["workers"]) if row["workers"] != "" else 0
        current_median = method_medians.get(
            (size, str(row["method"]), worker_key)
        )
        speedup = ""
        if (
            row["status"] == "ok"
            and baseline is not None
            and current_median is not None
        ):
            speedup = f"{baseline / current_median:.6f}"
        row["speedup_vs_sequential"] = speedup

        if row["method"] == "distributed":
            workers = int(row["workers"])
            one_worker = distributed_medians.get((size, 1))
            this_worker = distributed_medians.get((size, workers))
            if (
                row["status"] == "ok"
                and one_worker is not None
                and this_worker is not None
            ):
                strong_speedup = one_worker / this_worker
                row["strong_speedup_vs_one_worker"] = f"{strong_speedup:.6f}"
                row["parallel_efficiency"] = f"{strong_speedup / workers:.6f}"
            else:
                row["strong_speedup_vs_one_worker"] = ""
                row["parallel_efficiency"] = ""
        else:
            row["strong_speedup_vs_one_worker"] = ""
            row["parallel_efficiency"] = ""

    raw_path = ROOT / "results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for row in pending:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in FIELDS},
                    "end_to_end_ms": (
                        f"{float(row['end_to_end_ms']):.6f}"
                        if row.get("end_to_end_ms") not in (None, "")
                        else ""
                    ),
                    "memory_value_mib": (
                        f"{float(row['memory_value_mib']):.6f}"
                        if row.get("memory_value_mib") not in (None, "")
                        else ""
                    ),
                    "speedup_vs_sequential": row.get("speedup_vs_sequential", ""),
                }
            )

    write_summary(pending)
    environment = {
        "assignment": 4,
        "run_timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "cpu_model": cpu_model_name(),
        "physical_cores": physical_core_count(),
        "logical_processors": os.cpu_count(),
        "ram_total_gib": round(psutil.virtual_memory().total / (1024**3), 3),
        "physical_machines": 1,
        "execution_scope": (
            "Local Dask Distributed cluster on one physical machine; does not "
            "measure inter-machine network or aggregate multi-machine memory."
        ),
        "local_multiworker_permission": (
            "Not asserted. Multi-worker runs require instructor approval and "
            "the explicit --instructor-permission option."
        ),
        "framework": "Dask Distributed LocalCluster",
        "dask_version": __import__("dask").__version__,
        "distributed_version": __import__("distributed").__version__,
        "numpy_version": np.__version__,
        "threadpoolctl_version": __import__("threadpoolctl").__version__,
        "sizes": sizes,
        "workers_tested": worker_counts,
        "worker_processes": worker_counts,
        "threads_per_worker": THREADS_PER_WORKER,
        "threaded_reference_threads": 4,
        "affinity": "OS scheduler default; no CPU affinity pinning",
        "block_size": BLOCK_SIZE,
        "seed": SEED,
        "input_generator": "Python random.Random; integer-valued float64 in [-8, 8]",
        "repetitions": REPETITIONS,
        "warmups_per_configuration": WARMUPS,
        "task_wait_timeout_seconds": TIME_LIMIT_SECONDS,
        "memory_limit_mib": MEMORY_LIMIT_MIB,
        "timing_boundary": (
            "Distributed end-to-end: cluster/client startup, matrix partitioning, "
            "input scatter, task submission, wait, result gather and "
            "cluster shutdown, excluding the separate worker-memory sampling RPC. "
            "Input generation and correctness validation are excluded. "
            "Sequential/threaded rows time the method call only; memory tracing "
            "uses an additional untimed invocation."
        ),
        "transfer_measurements": (
            "Input scatter and result gather wall times plus logical NumPy payload "
            "sizes; these are not isolated network-only measurements."
        ),
        "internal_library_threads": 1,
        "internal_thread_controls": {
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        },
        "worker_threadpool_settings": distributed_configs,
        "memory_measurement": (
            "Sequential/threaded: peak tracemalloc allocations. Distributed: "
            "sampled process RSS in workers after result collection. Metrics have "
            "different scopes and must not be treated as directly equivalent."
        ),
        "distributed_aggregation": (
            "Each output tile is a sum of products of matching A and B tiles; "
            "Dask futures schedule the tile products and reductions."
        ),
    }
    (ROOT / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    return {
        "raw_measurements": len(pending),
        "configurations": len(grouped),
        "distributed_configurations": len(distributed_configs),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = {}
    for row in rows:
        workers = int(row["workers"]) if row["workers"] != "" else 0
        grouped.setdefault(
            (int(row["size"]), str(row["method"]), workers), []
        ).append(row)

    fields = [
        "size",
        "method",
        "workers",
        "worker_processes",
        "physical_machines",
        "median_end_to_end_ms",
        "iqr_end_to_end_ms",
        "median_compute_wait_ms",
        "median_setup_ms",
        "median_input_scatter_ms",
        "median_result_collection_ms",
        "median_worker_rss_max_sampled_mib",
        "median_worker_rss_total_sampled_mib",
        "median_memory_value_mib",
        "memory_metric",
        "samples",
        "status",
        "speedup_vs_sequential",
        "strong_speedup_vs_one_worker",
        "parallel_efficiency",
    ]
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key, group in grouped.items():
            good = [
                row
                for row in group
                if row["status"] == "ok"
                and row.get("end_to_end_ms") not in (None, "")
            ]
            values = [float(row["end_to_end_ms"]) for row in good]
            representative = good[-1] if good else group[-1]

            def median_field(field: str) -> str:
                field_values = [
                    float(row[field])
                    for row in good
                    if row.get(field) not in (None, "")
                ]
                return (
                    f"{statistics.median(field_values):.6f}"
                    if field_values
                    else ""
                )

            writer.writerow(
                {
                    "size": key[0],
                    "method": key[1],
                    "workers": key[2] if key[1] == "distributed" else "",
                    "worker_processes": representative.get("worker_processes", ""),
                    "physical_machines": representative.get("physical_machines", 1),
                    "median_end_to_end_ms": (
                        f"{statistics.median(values):.6f}" if values else ""
                    ),
                    "iqr_end_to_end_ms": (
                        f"{calculate_iqr(values):.6f}" if values else ""
                    ),
                    "median_compute_wait_ms": median_field("compute_wait_ms"),
                    "median_setup_ms": median_field("setup_ms"),
                    "median_input_scatter_ms": median_field("input_scatter_ms"),
                    "median_result_collection_ms": median_field(
                        "result_collection_ms"
                    ),
                    "median_worker_rss_max_sampled_mib": median_field(
                        "worker_rss_max_sampled_mib"
                    ),
                    "median_worker_rss_total_sampled_mib": median_field(
                        "worker_rss_total_sampled_mib"
                    ),
                    "median_memory_value_mib": median_field("memory_value_mib"),
                    "memory_metric": representative.get("memory_metric", ""),
                    "samples": f"{len(values)}/{len(group)}",
                    "status": (
                        "ok"
                        if all(row["status"] == "ok" for row in group)
                        else "mixed: " + ", ".join(sorted({
                            str(row["status"])
                            for row in group
                            if row["status"] != "ok"
                        }))
                    ),
                    "speedup_vs_sequential": representative.get(
                        "speedup_vs_sequential", ""
                    ),
                    "strong_speedup_vs_one_worker": representative.get(
                        "strong_speedup_vs_one_worker", ""
                    ),
                    "parallel_efficiency": representative.get(
                        "parallel_efficiency", ""
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--workers", nargs="+", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--instructor-permission",
        action="store_true",
        help="confirm instructor approval for local execution with multiple workers",
    )
    args = parser.parse_args()
    if any(n <= 0 for n in args.sizes) or any(w <= 0 for w in args.workers):
        parser.error("sizes and worker counts must be positive")
    if any(worker > 1 for worker in args.workers) and not args.instructor_permission:
        parser.error(
            "local execution with multiple workers requires instructor approval; "
            "after approval, pass --instructor-permission"
        )
    print(json.dumps(run_benchmark(args.sizes, args.workers)))


if __name__ == "__main__":
    main()
