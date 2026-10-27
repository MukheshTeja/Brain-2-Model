"""Loaders for the real fMRI data (Phase B). Run scripts/download_data.py first.

ds003076 (deductive reasoning): trial-wise GLMsingle betas released by Xiao
et al. 2026 -- one response vector per reasoning problem per subject.
Row order (verified against Xiao et al.'s loader, utils.get_all_fmri_data_latest):
sorted runs Syllogisms_run-01, Syllogisms_run-02, Transitive_run-01,
Transitive_run-02, 18 problems each, in presentation order. Rows 36+7 and
36+13 are logically contradictory transitive items (A>B, B>C, C>A) and are
dropped, leaving 36 syllogisms + 34 transitive problems.

IMPORTANT: Xiao et al. selected the top-10% voxels SEPARATELY for the two
tasks, so columns mean different voxels for syllogism rows vs transitive
rows. Every analysis here therefore treats the two tasks as separate blocks
(separate PCA, separate RDM), never mixing their columns.

HCP relational processing: trial-wise betas (MD network) for 24 trials per
subject; the stimulus lists come in two versions, so items are matched by
stimulus file name across subjects.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from .reasoning_task import Problem, evaluate_problem, parse_fmri_problem

NETWORKS = ("deductive_reasoning", "md", "language")
CONTRADICTORY_ROWS = (36 + 7, 36 + 13)
TASKS = ("syllogism", "transitive")


@dataclass
class DS003076:
    problems: list[Problem]  # 70, aligned with rows of every beta matrix
    table: pd.DataFrame  # per-problem metadata (task, trial_type, text, mean RT/acc)
    betas: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # network -> subject -> [70, V]
    subjects: list[str] = field(default_factory=list)

    def task_rows(self, task: str) -> np.ndarray:
        return np.where(self.table["task"].to_numpy() == task)[0]


def load_ds003076(data_dir: str | Path = "data", networks=NETWORKS) -> DS003076:
    data_dir = Path(data_dir)
    items = pickle.load(open(data_dir / "ds003076_info" / "task_items.pkl", "rb"))
    events = pickle.load(open(data_dir / "ds003076_info" / "events.pkl", "rb"))
    beta_root = data_dir / "ds003076_betas"
    subjects = sorted(p.name for p in (beta_root / f"all_extracted_beta_{networks[0]}").iterdir() if p.is_dir())

    rows, problems = [], []
    for run in sorted(items):
        df = items[run].reset_index(drop=True)
        for i, r in df.iterrows():
            rows.append({"run": run, "pos": i, "task": "syllogism" if "Syllogisms" in run else "transitive",
                         "trial_type": r.trial_type,
                         "text": " | ".join([r.premise1, r.premise2, r.premise3, "=> " + r.conclusion])})
    keep = [i for i in range(len(rows)) if i not in CONTRADICTORY_ROWS]
    table = pd.DataFrame([rows[i] for i in keep]).reset_index(drop=True)

    rt = np.full((len(subjects), len(rows)), np.nan)
    acc = np.full((len(subjects), len(rows)), np.nan)
    for si, s in enumerate(subjects):
        r, a = [], []
        for run in sorted(items):
            ev = events[s][run]
            r += list(ev.response_time)
            a += list(ev.accuracy)
        rt[si], acc[si] = r, a
    table["mean_rt"] = np.nanmean(rt[:, keep], axis=0)
    table["human_acc"] = np.nanmean(acc[:, keep], axis=0)

    for i in keep:
        run_df = items[rows[i]["run"]].reset_index(drop=True)
        r = run_df.iloc[rows[i]["pos"]]
        problems.append(parse_fmri_problem([r.premise1, r.premise2, r.premise3], r.conclusion, r.trial_type, seed=i))
    table["label"] = [p.label for p in problems]
    table["n_required"] = [p.n_required for p in problems]
    table["negated"] = [int(p.negated) for p in problems]

    ds = DS003076(problems=problems, table=table, subjects=subjects)
    for net in networks:
        ds.betas[net] = {}
        for s in subjects:
            x = sio.loadmat(beta_root / f"all_extracted_beta_{net}" / s / f"{s}.mat")["all_voxels"].astype(np.float64)
            x = x[keep]
            x = x[:, ~np.isnan(x).any(axis=0)]
            ds.betas[net][s] = x
    return ds


def _onehots(columns: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([(v[:, None] == np.unique(v)[None, :]).astype(float) for v in columns], axis=1)


def design_labels(ds: DS003076, task: str) -> np.ndarray:
    """Main-effect one-hots of the logical-structure variables the knockout
    erases for one task block: validity, premises required, conclusion
    polarity, and fact polarity (syllogisms) / comparison direction
    (transitive). Main effects only (rank ~4), so the knockout removes the
    task-design subspace without deleting most of a 20-dim signal."""
    rows = ds.task_rows(task)
    t = ds.table.iloc[rows]
    probs = [ds.problems[i] for i in rows]
    if task == "syllogism":
        fourth = np.array([int(p.premises[-1][0] == "ISNOT") for p in probs])
    else:
        fourth = []
        for p in probs:  # is the conclusion's first-named person truly "more"? (a > b in the order)
            more = Problem("transitive", p.premises, ("MORE", p.conclusion[1], p.conclusion[2]), 0, 0, False)
            fourth.append(evaluate_problem(more))
        fourth = np.array(fourth)
    return _onehots([t["label"].to_numpy(), t["n_required"].to_numpy(), t["negated"].to_numpy(), fourth])


# ---------------------------------------------------------------- HCP
def hcp_blocks(h: "HCPRelational", min_subjects: int = 5) -> dict[str, dict]:
    """Group HCP subjects by stimulus list (two lists in this release). Each
    list becomes one alignment block: its items + each subject's betas."""
    groups: dict[tuple, list[str]] = {}
    for sid, b in h.betas.items():
        groups.setdefault(tuple(np.where(~np.isnan(b[:, 0]))[0]), []).append(sid)
    out = {}
    for k, (items, sids) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        if len(sids) < min_subjects:
            continue
        idx = np.array(items)
        out[f"list{k + 1}"] = {"items": idx, "subjects": sorted(sids),
                               "betas": [h.betas[s][idx] for s in sorted(sids)],
                               "design": _onehots([h.item_table["answer"].to_numpy()[idx],
                                                   h.item_table["dimension"].to_numpy()[idx]])}
    return out


@dataclass
class HCPRelational:
    items: list[str]  # stimulus file names (union across stimulus lists)
    betas: dict[str, np.ndarray]  # subject -> [n_items, V] (NaN rows for items the subject never saw)
    item_table: pd.DataFrame


def load_hcp_relational(data_dir: str | Path = "data") -> HCPRelational:
    import json

    root = Path(data_dir) / "hcp_relational"
    per_subject = {}
    for mat in sorted((root / "betas").glob("*/*.mat")):
        sid = mat.stem
        info = pd.read_csv(mat.with_name(f"{sid}_trialinfo.csv"))
        x = sio.loadmat(mat)["all_voxels"].astype(np.float64)
        x = x[:, ~np.isnan(x).any(axis=0)]
        # average repeated presentations of the same stimulus
        df = pd.DataFrame(x)
        df["stim"] = info["Stimulus"].to_numpy()
        per_subject[sid] = df.groupby("stim").mean()
    items = sorted(set().union(*[set(d.index) for d in per_subject.values()]))
    betas = {}
    for sid, d in per_subject.items():
        full = np.full((len(items), d.shape[1]), np.nan)
        idx = [items.index(s) for s in d.index]
        full[idx] = d.to_numpy()
        betas[sid] = full
    mapping = {}
    for line in open(root / "stimuli_mapping_final.jsonl"):
        rec = json.loads(line)
        mapping[rec["stimulus_file"]] = rec
    rows = []
    for s in items:
        # e.g. int_load_inhib_NO_SHA.6.bmp: top pair differs in SHApe, answer NO
        parts = s.split(".")[0].split("_")
        rows.append({"stim": s, "answer": int(parts[-2] == "YES"), "dimension": parts[-1],
                     "load": int(parts[1] == "load"), "inhib": int(parts[2] == "inhib"),
                     "positions": mapping.get(s, {}).get("positions")})
    return HCPRelational(items=items, betas=betas, item_table=pd.DataFrame(rows))
