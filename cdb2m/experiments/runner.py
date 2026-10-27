"""Tiny, dependency-free parallel job runner with on-disk resume.

Every job is a plain dict with a unique `job_id`; each finished job's rows
are appended to a CSV immediately, so an interrupted run (laptop sleep,
Ctrl-C) resumes where it stopped instead of starting over. Workers are
single-threaded torch processes (`spawn` start method, identical on
Windows / macOS / Linux), so N workers use N CPU cores and no GPU.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def default_workers() -> int:
    """Leave one core free and cap at 6: each worker is a separate Python +
    torch process (~350 MB with CPU-only torch, ~0.9 GB with a CUDA build),
    so 6 workers stay well inside a 16 GB laptop."""
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def parallel_map(fn: Callable, items: list, n_workers: int) -> list:
    if n_workers <= 1 or len(items) <= 1:
        _init_worker()
        return [fn(x) for x in items]
    with mp.get_context("spawn").Pool(min(n_workers, len(items)), initializer=_init_worker) as pool:
        return pool.map(fn, items)


def run_jobs(jobs: list[dict], fn: Callable[[dict], list[dict]], out_csv: str | Path, n_workers: int,
             label: str = "jobs") -> pd.DataFrame:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        done = set(pd.read_csv(out_csv, usecols=["job_id"])["job_id"].astype(str))
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"[{label}] {len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run on {n_workers} worker(s)", flush=True)
    if todo:
        t0 = time.time()
        header_needed = not (out_csv.exists() and out_csv.stat().st_size > 0)
        with open(out_csv, "a", newline="") as fh:
            writer = None
            if n_workers <= 1:
                _init_worker()
                iterator = map(fn, todo)
                pool = None
            else:
                pool = mp.get_context("spawn").Pool(min(n_workers, len(todo)), initializer=_init_worker)
                iterator = pool.imap_unordered(fn, todo)
            try:
                for i, rows in enumerate(iterator, 1):
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                        if header_needed:
                            writer.writeheader()
                    writer.writerows(rows)
                    fh.flush()
                    if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                        el = time.time() - t0
                        print(f"[{label}] {i}/{len(todo)} done, {el / 60:.1f} min elapsed, "
                              f"~{el / i * (len(todo) - i) / 60:.1f} min left", flush=True)
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
    return pd.read_csv(out_csv)
