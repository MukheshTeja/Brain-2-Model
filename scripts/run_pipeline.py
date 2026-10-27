"""One command for the whole CD-B2M pipeline.

    python -m scripts.download_data                 # once, ~24 MB
    python -m scripts.run_pipeline --preset quick   # ~10-15 min on a 8-core laptop
    python -m scripts.run_pipeline --preset full    # the deck's protocol (10 seeds), ~30-45 min on 8 cores
    python -m scripts.make_report                   # figures + REPORT.md from the CSVs

CPU only (no GPU needed), < 1 GB RAM, ~30 MB disk for data + results.
Interrupted runs resume from results/*.csv automatically.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cdb2m.experiments import phase_a, phase_b
from cdb2m.experiments.runner import default_workers, parallel_map, run_jobs


def presets(name: str):
    a, b = phase_a.PhaseASettings(), phase_b.PhaseBSettings()
    if name == "quick":
        a = replace(a, seeds=tuple(range(4)), seeds_other_students=tuple(range(3)), rq2_seeds=tuple(range(2)),
                    dose_alphas=(0.1, 0.2, 0.4))
        b = replace(b, seeds=tuple(range(4)), seeds_other_students=tuple(range(3)), rq2_seeds=tuple(range(2)),
                    dose_alphas=(0.1, 0.2, 0.4), n_eval=600)
    elif name == "smoke":  # seconds-scale correctness check
        a = replace(a, seeds=(0,), seeds_other_students=(0,), rq2_seeds=(0,), dose_alphas=(0.2,), n_train_epochs=20,
                    teacher_train_steps=30, n_eval_sequences=20)
        b = replace(b, seeds=(0,), seeds_other_students=(0,), rq2_seeds=(0,), dose_alphas=(0.2,), n_steps=20,
                    teacher_steps=30, n_eval=50)
    elif name != "full":
        raise ValueError(name)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=["smoke", "quick", "full"])
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--phase", default="AB", help="A, B or AB")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sa, sb = presets(args.preset)
    sb = replace(sb, data_dir=args.data_dir)
    tag = "" if args.preset == "full" else f"_{args.preset}"
    t0 = time.time()

    if "A" in args.phase:
        print("=== Phase A: synthetic brain sandbox ===", flush=True)
        seeds = sorted(set(sa.seeds) | set(sa.seeds_other_students) | set(sa.rq2_seeds))
        ctxs = parallel_map(phase_a.build_context, [(sa, s) for s in seeds], args.workers)
        contexts = {c["seed"]: c for c in ctxs}
        print(f"  built {len(contexts)} seed contexts in {(time.time() - t0) / 60:.1f} min", flush=True)
        run_jobs(phase_a.make_jobs(sa, contexts), phase_a.execute_job, out / f"phaseA{tag}.csv", args.workers, "Phase A")

    if "B" in args.phase:
        print("=== Phase B: real fMRI ===", flush=True)
        t1 = time.time()
        shared = phase_b.shared_context(sb)
        seeds = sorted(set(sb.seeds) | set(sb.seeds_other_students) | set(sb.rq2_seeds))
        ctxs = parallel_map(phase_b.build_context, [(sb, s, shared) for s in seeds], args.workers)
        contexts = {c["seed"]: c for c in ctxs}
        keys = sorted({k for c in ctxs for k in c["reliability"]})  # coalition keys exist only for RQ2 seeds
        rel = {k: float(np.mean([c["reliability"][k] for c in ctxs if k in c["reliability"]])) for k in keys}
        meta = {"reliability": rel, "realsnr_noise_scale": shared.get("realsnr_noise", {})}
        (out / f"phaseB{tag}_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"  built {len(contexts)} seed contexts in {(time.time() - t1) / 60:.1f} min", flush=True)
        run_jobs(phase_b.make_jobs(sb, contexts), phase_b.execute_job, out / f"phaseB{tag}.csv", args.workers, "Phase B")
    print(f"=== done in {(time.time() - t0) / 60:.1f} min -> now run: python -m scripts.make_report "
          f"--preset {args.preset} ===", flush=True)


if __name__ == "__main__":
    main()
