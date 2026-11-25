# Assignment 4 — Distributed Matrix Multiplication

## Study and scope

This study reuses the validated `i-j-k` sequential method and the
row-partitioned `ThreadPoolExecutor` method from Assignment 3. It adds a
block-partitioned implementation using Dask Distributed. Each `A[i,k]` and
`B[k,j]` tile is scattered to Dask workers; workers compute `A[i,k] @ B[k,j]`
and Dask reduces the partial products into output tiles. The result is gathered
to the client.

The current measurements use a one-worker Dask cluster on one physical
machine. They do **not** measure inter-machine networking or multi-machine
memory. The assignment PDF permits a local execution with multiple workers
only if the instructor approves it. The benchmark therefore refuses to start
multiple local workers unless `--instructor-permission` is explicitly passed.

## Files

- `methods.py`: reused sequential/threaded methods and distributed tile logic.
- `benchmark_assignment4.py`: benchmark, raw results, summaries and metadata.
- `generate_graphics.py`: runtime, speedup and worker-memory plots.
- `generate_report.py`: render the report and figures into a PDF.
- `requirements.txt`: pinned Python dependencies.
- `results.csv`, `summary.csv`, `environment.json`: measurements and settings.
- `REPORT.md`, `Assignment04_Javier_Guerra_Report.pdf`: report.

## Install and run

Use Python 3.10 or later. From this directory:

```powershell
python -m pip install -r requirements.txt
python methods.py
python benchmark_assignment4.py
python generate_graphics.py
python generate_report.py
```

By default, the benchmark runs sizes `32, 64, 128, 256`, one local Dask worker
process, one thread per worker, five measured repetitions and one warm-up.
Inputs are generated with seed `20261124`, with integer-valued float64 entries
in `[-8, 8]`. All three methods use the same matrices for each size.

Only after receiving instructor approval for local multiprocess execution, run
the strong-scaling sweep with:

```powershell
python benchmark_assignment4.py --workers 1 2 4 --instructor-permission
python generate_graphics.py
```

This keeps each global matrix size fixed while increasing the number of Dask
worker processes. These workers remain on one physical machine; they are not
nodes. The flag is a deliberate confirmation, not evidence that permission has
been granted.

## Measurement and interpretation

Distributed end-to-end time includes cluster/client startup, input
partitioning, scattering the input tiles, task submission, waiting for task
completion, gathering the result and cluster shutdown. Input generation and
correctness checking are excluded. Setup, scatter, wait and gather are also
recorded separately. These phase durations overlap in meaning with
end-to-end time and must not be added as independent costs. Scatter/gather
durations and payload sizes are not isolated network measurements.

The sequential and threaded rows time the multiplication method call only, so
their speedup is a contextual baseline rather than a perfectly matched
end-to-end comparison. `summary.csv` reports median and IQR. Distributed
strong-scaling speedup and efficiency compare each worker count with the
one-worker Dask run at the same matrix size.

Sequential/threaded memory is measured with `tracemalloc`. Distributed memory
is sampled worker-process RSS after result collection. They have different
measurement scopes and should not be treated as equivalent. Hardware,
software, affinity and timing settings are in `environment.json`.
