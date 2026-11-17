"""Reproducible laptop experiment: validation selection, then a fresh null ladder.

python -m scripts.run_laptop --stage select
python -m scripts.run_laptop --stage evaluate --workers 2
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time
from itertools import permutations
# Load pandas before torch to avoid the host conda pyarrow/torch DLL-order conflict.
import numpy as np
import pandas as pd
import torch
from cdb2m.fmri import reasoning_task as rt
from cdb2m.fmri.datasets import load_ds003076, design_labels
from cdb2m.fmri.brain_teacher import build_block_arm
from cdb2m.training.laptop import fit, score, problem_key, split_problems, partition
from scripts.audit_data import audit

VERSION = 1


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def source_hash():
    digest = hashlib.sha256()
    for directory in ("cdb2m", "nullladder"):
        for p in sorted(Path(directory).rglob("*.py")):
            digest.update(p.as_posix().encode()); digest.update(p.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def evaluate(model, excluded, n):
    result = {}
    for name, size, families, shuffle in [
        ("iid",3,("syllogism","transitive"),False),
        ("premises_4",4,("syllogism","transitive"),False),
        ("premises_5",5,("syllogism","transitive"),False),
        ("premises_6",6,("syllogism","transitive"),False),
        ("shuffled_order",3,("syllogism","transitive"),True),
        ("propositional",2,("propositional",),False)]:
        # Propositional space is only 360 encodings (4 kinds x 90 pairs).
        if name == "propositional":
            all_props = [rt._propositional(pair,kind) for pair in permutations(range(rt.N_SLOTS),2)
                         for kind in ("MP","AC","MT","DA")]
            probs = [p for p in all_props if partition(p)=="test" and problem_key(p) not in excluded]
        else:
            probs = split_problems(n,90_000_000+size,"test",size,families,excluded)
        batch = rt.problems_to_batch(probs, seed=91_000_000, shuffle_premises=shuffle)
        accs = []
        for start in range(0,len(probs),128):
            part = {k:v[start:start+128] for k,v in batch.items()}
            accs.append((len(part["inputs"]),score(model,part)[0]))
        result[name] = {"accuracy":sum(w*a for w,a in accs)/len(probs),"n":len(probs)}
    return result


def job(spec):
    torch.set_num_threads(1)
    ds = load_ds003076(spec["data_dir"], networks=("deductive_reasoning",))
    recorded = rt.problems_to_batch(ds.problems)
    blocks = None
    if spec["rung"] != "baseline":
        blocks = {}
        for task in ("syllogism", "transitive"):
            rows = ds.task_rows(task)
            arm = build_block_arm([ds.betas["deductive_reasoning"][s][rows] for s in ds.subjects],
                                  design_labels(ds,task),seed=spec["seed"])
            blocks[task] = (torch.as_tensor(rows),torch.tensor(arm.rung_embeddings[spec["rung"]],dtype=torch.float32))
    model, info = fit(spec["model"],spec["seed"],recorded,blocks,loss_name=spec["loss"],
                      alpha=spec["alpha"],steps=spec["steps"],legacy=spec.get("legacy",False))
    path = Path(spec["out"])/spec["stage"]/spec["id"]
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint = path.with_suffix(".pt")
    tmp = checkpoint.with_suffix(".pt.tmp")
    torch.save({"model_state":model.state_dict(),"model_kind":spec["model"],"spec":spec},tmp)
    tmp.replace(checkpoint)
    result = {"spec":spec,**info,"checkpoint":str(checkpoint)}
    if spec["stage"] == "evaluate":
        result["test"] = evaluate(model,{problem_key(p) for p in ds.problems},spec["n_eval"])
    try:
        import psutil
        mem = psutil.Process().memory_info()
        result["peak_process_mb"] = getattr(mem,"peak_wset",mem.rss)/1e6
    except ImportError:
        result["peak_process_mb"] = None
    atomic_json(path.with_suffix(".json"),result)
    return {"id":spec["id"],"val_accuracy":info["validation_accuracy"],"seconds":round(info["seconds"],1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["select","evaluate"],default="select")
    ap.add_argument("--out",default="results/laptop")
    ap.add_argument("--data-dir",default="data")
    ap.add_argument("--workers",type=int,choices=[1,2],default=1)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--n-eval",type=int,default=1000)
    args = ap.parse_args()
    if args.steps < 1 or args.n_eval < 1:
        ap.error("steps and n-eval must be positive")
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data = audit(args.data_dir); atomic_json(out/"data_audit.json",data)
    if not data["ok"]:
        raise RuntimeError(data["errors"])
    config = {"version":VERSION,"source_sha256":source_hash(),"steps":args.steps,"n_eval":args.n_eval,
              "data_manifest_sha256":hashlib.sha256((Path(args.data_dir)/"MANIFEST.json").read_bytes()).hexdigest(),
              "torch":torch.__version__,"numpy":np.__version__,"python":platform.python_version(),
              "selection_seeds":[0,1],"evaluation_seeds":[20,21,22,23,24],
              "split":"blake2b exact statement encoding 80/10/10; recorded examples excluded from validation/test",
              "selection_metric":"mean IID validation cross entropy; no OOD test selection"}
    path = out/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Protocol/source changed. Use a new --out directory to prevent stale result reuse.")
    atomic_json(path,config)
    jobs = []
    def add(model,seed,rung="baseline",loss="linear_mse",alpha=.2,legacy=False):
        ident = f"{model}_{'legacy' if legacy else loss}_{rung}_s{seed}"
        spec = dict(id=ident,model=model,seed=seed,rung=rung,loss=loss,alpha=alpha,legacy=legacy,
                    steps=args.steps,n_eval=args.n_eval,data_dir=args.data_dir,out=args.out,stage=args.stage)
        if not (out/args.stage/f"{ident}.json").exists():
            jobs.append(spec)
    if args.stage == "select":
        for model in ("gru32","gru64","relational32"):
            for seed in (0,1):
                add(model,seed)
    else:
        selection = json.loads((out/"selection.json").read_text())
        for seed in config["evaluation_seeds"]:
            add("gru32",seed,legacy=True)
            for rung in ("baseline","R0","R1","R2","R3","R4","R5","knockout"):
                add(selection["model"],seed,rung)
            # Local loss ablations, explicitly not reproductions of LLM papers.
            for loss in ("cka","infonce"):
                add(selection["model"],seed,"R5",loss)
    print(f"{args.stage}: {len(jobs)} unfinished jobs, {args.workers} CPU worker(s)",flush=True)
    if args.workers == 1:
        for j in jobs:
            print(job(j),flush=True)
    elif jobs:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for res in pool.imap_unordered(job,jobs):
                print(res,flush=True)
    if args.stage == "select":
        results = [json.loads(p.read_text()) for p in (out/"select").glob("*.json")]
        means = {model:float(np.mean([r["validation_ce"] for r in results if r["spec"]["model"]==model]))
                 for model in ("gru32","gru64","relational32")}
        selected = {"model":min(means,key=means.get),"mean_validation_ce":means,
                    "note":"Selected before reading final-test results. Brain weight fixed at 0.2 for every ladder arm."}
        atomic_json(out/"selection.json",selected); print(selected,flush=True)


if __name__ == "__main__":
    main()
