"""Statistics behind the CD-B2M verdicts. Operates on the tidy CSVs written
by cdb2m/experiments (one row per job x evaluation variant).

Conventions
- `value` = accuracy on one evaluation variant. "zero-shot score" = mean over
  the UNSEEN variants (everything except "iid") for one run.
- gain = run score - the alpha=0 baseline score of the SAME student, seed
  and variant (paired design: same init, same data stream, same eval sets).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from nullladder.equivalence import tost_equivalence
from nullladder.shapley import exact_shapley

LADDER = ["R0", "R1", "R2", "R3", "R4", "R5"]


def add_gains(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the paired alpha=0 baseline. Baseline jobs emit one row-set per
    arm (with that arm's CKA targets); arms without their own baseline rows
    (RQ2 coalitions) use the same student/seed/dataset baseline."""
    df = df.copy()
    df["_bds"] = df["dataset"].replace({"sandbox_rq2": "sandbox_main"})
    keys = ["phase", "_bds", "student", "seed", "variant"]
    base = df[df.rung == "baseline"].groupby(keys)["value"].first().rename("baseline_value").reset_index()
    out = df.merge(base, on=keys, how="left")
    out["gain"] = out["value"] - out["baseline_value"]
    return out.drop(columns="_bds")


def zero_shot(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """One row per run: mean over unseen variants (+ the iid score alongside)."""
    keys = ["phase", "dataset", "student", "arm", "rung", "alpha", "seed"]
    zs = df[df.variant != "iid"].groupby(keys).agg(score=(col, "mean"), cka_R5=("cka_R5", "first"),
                                                   cka_R3=("cka_R3", "first"), cka_rung=("cka_rung", "first"))
    iid = df[df.variant == "iid"].groupby(keys)[col].mean().rename("iid")
    return zs.join(iid).reset_index()


def paired(runs: pd.DataFrame, a: str, b: str, col: str = "score") -> tuple[np.ndarray, np.ndarray]:
    pa = runs[runs.rung == a].set_index("seed")[col]
    pb = runs[runs.rung == b].set_index("seed")[col]
    common = sorted(set(pa.index) & set(pb.index))
    return pa.loc[common].to_numpy(), pb.loc[common].to_numpy()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    return float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan")


def p_greater(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, y):
        return float("nan")
    return float(stats.ttest_rel(x, y, alternative="greater").pvalue)


@dataclass
class Verdict:
    label: str
    clause1_r5_beats_all: bool
    clause2_knockout_eq_r3: bool
    r5_eq_r3: bool
    r5_eq_r4: bool
    details: dict


def verdict(runs: pd.DataFrame, thr: dict) -> Verdict:
    """The deck's verdict rule on one (dataset, student, arm) slice at alpha_main.

    GENUINE TRANSFER iff R5 beats every rung R0-R4 (means; R5-R4 paired d >
    min_effect_size_r5_vs_r4) AND knockout is TOST-equivalent to R3.
    REGULARIZER iff R5 is TOST-equivalent to R3 (the max-entropy null with all
    of the brain's 1st/2nd-order statistics does as well as the real brain).
    INCONCLUSIVE otherwise (e.g. underpowered: neither difference nor equivalence)."""
    means = runs.groupby("rung")["score"].mean()
    det = {"means": means.to_dict()}
    beats = all(means.get("R5", np.nan) > means.get(r, np.inf) for r in LADDER[:-1])
    x5, x4 = paired(runs, "R5", "R4")
    d54 = cohens_d(x5, x4)
    det["d_R5_R4"] = d54
    det["p_R5_gt"] = {r: p_greater(*paired(runs, "R5", r)) for r in LADDER[:-1]}
    clause1 = bool(beats and d54 > thr["min_effect_size_r5_vs_r4"])
    ko = paired(runs, "knockout", "R3")
    clause2 = False
    if len(ko[0]) > 1:
        t = tost_equivalence(*ko, margin=thr["knockout_vs_r3_margin"])
        clause2, det["tost_ko_r3"] = bool(t.equivalent), (t.mean_diff, t.p_tost)
    r53 = paired(runs, "R5", "R3")
    t53 = tost_equivalence(*r53, margin=thr["knockout_vs_r3_margin"]) if len(r53[0]) > 1 else None
    r54 = paired(runs, "R5", "R4")
    t54 = tost_equivalence(*r54, margin=thr["hollow_r5_vs_r4_margin"]) if len(r54[0]) > 1 else None
    det["tost_r5_r3"] = (t53.mean_diff, t53.p_tost) if t53 else None
    det["tost_r5_r4"] = (t54.mean_diff, t54.p_tost) if t54 else None
    eq53 = bool(t53 and t53.equivalent)
    eq54 = bool(t54 and t54.equivalent)
    if clause1 and clause2:
        label = "GENUINE TRANSFER"
    elif eq53:
        label = "REGULARIZER"
    elif clause1:
        label = "TRANSFER (knockout incomplete)"
    else:
        label = "INCONCLUSIVE"
    return Verdict(label, clause1, clause2, eq53, eq54, det)


def fingerprint(df_gain: pd.DataFrame, rungs=("R0", "R3", "R4", "R5", "knockout")) -> pd.DataFrame:
    """Mean gain per evaluation variant per rung + R5-vs-R3 paired p per variant."""
    rows = []
    for v, g in df_gain.groupby("variant"):
        rec = {"variant": v}
        for r in rungs:
            rec[r] = g[g.rung == r]["gain"].mean()
        a = g[g.rung == "R5"].set_index("seed")["gain"]
        b = g[g.rung == "R3"].set_index("seed")["gain"]
        c = sorted(set(a.index) & set(b.index))
        rec["R5-R3"] = float((a.loc[c] - b.loc[c]).mean()) if c else np.nan
        rec["p(R5>R3)"] = p_greater(a.loc[c].to_numpy(), b.loc[c].to_numpy()) if len(c) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- RQ1
def mediation(runs: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """alpha = randomised dose (treatment T), CKA-to-real-brain = mediator M,
    CKA-to-R3 surrogate = covariate C, zero-shot gain = outcome Y.
        M = a0 + a*T + c1*C + e1
        Y = b0 + b*M + d*T + c2*C + e2
    ACME = a*b (indirect effect of dose through brain similarity), ADE = d.
    Percentile CI from a seed-clustered bootstrap. Sensitivity: the residual
    correlation rho at which ACME would vanish (Imai, Keele & Yamamoto 2010,
    linear-model closed form)."""
    d = runs.dropna(subset=["cka_R5", "cka_R3", "gain"]).copy()
    if len(d) < 8 or d["alpha"].nunique() < 3:
        return {}
    for c in ("alpha", "cka_R5", "cka_R3", "gain"):
        d[c] = d[c].astype(float)

    def fit(dd):
        X1 = np.c_[np.ones(len(dd)), dd.alpha, dd.cka_R3]
        a_coef, *_ = np.linalg.lstsq(X1, dd.cka_R5, rcond=None)
        X2 = np.c_[np.ones(len(dd)), dd.cka_R5, dd.alpha, dd.cka_R3]
        b_coef, *_ = np.linalg.lstsq(X2, dd.gain, rcond=None)
        return a_coef[1], b_coef[1], b_coef[2], b_coef[3]

    a, b, ade, c2 = fit(d)
    seeds = d.seed.unique()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        dd = pd.concat([d[d.seed == s] for s in pick])
        try:
            aa, bb, dd_, _ = fit(dd)
            boots.append((aa * bb, dd_, bb))
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots)
    # sensitivity rho (linear SEM): ACME(rho) = a * (sigma1/sigma2) * (rho~ - rho*sqrt((1-rho~^2)/(1-rho^2)))
    X1 = np.c_[np.ones(len(d)), d.alpha, d.cka_R3]
    e1 = d.cka_R5 - X1 @ np.linalg.lstsq(X1, d.cka_R5, rcond=None)[0]
    e3 = d.gain - X1 @ np.linalg.lstsq(X1, d.gain, rcond=None)[0]
    rho_tilde = float(np.corrcoef(e1, e3)[0, 1])
    # ACME(rho)=0  <=>  rho = rho_tilde (closed form: zero exactly when rho equals rho~)
    pc = stats.pearsonr(e1, e3)
    return {"n_runs": len(d), "a_dose_to_cka": float(a), "b_cka_to_gain": float(b), "ACME": float(a * b),
            "ACME_ci": tuple(np.percentile(boots[:, 0], [2.5, 97.5])) if len(boots) else (np.nan, np.nan),
            "ADE": float(ade), "ADE_ci": tuple(np.percentile(boots[:, 1], [2.5, 97.5])) if len(boots) else (np.nan, np.nan),
            "b_ci": tuple(np.percentile(boots[:, 2], [2.5, 97.5])) if len(boots) else (np.nan, np.nan),
            "partial_r_cka_gain": float(pc[0]), "partial_p": float(pc[1]), "sensitivity_rho": rho_tilde,
            "cka_R3_coef": float(c2)}


# --------------------------------------------------------------------- RQ2
def region_shapley(runs: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Exact Shapley over regions. Two value functions:
       v_raw(S)       = mean zero-shot gain when aligning to real coalition S (R5)
       v_struct(S)    = mean[gain(R5_S) - gain(R3_S)]  (structure beyond brain statistics)
    v(empty) = 0. Seed-bootstrap CIs."""
    runs = runs.copy()
    players = sorted({p for a in runs.arm.unique() for p in a.split("+")})
    seeds = sorted(runs.seed.unique())

    def values(sub):
        g = sub.groupby(["arm", "rung"])["gain"].mean()
        v_raw = {frozenset(a.split("+")): g.get((a, "R5"), np.nan) for a in sub.arm.unique()}
        v_st = {frozenset(a.split("+")): g.get((a, "R5"), np.nan) - g.get((a, "R3"), np.nan) for a in sub.arm.unique()}
        v_raw[frozenset()] = 0.0
        v_st[frozenset()] = 0.0
        return (exact_shapley(players, lambda s: v_raw[frozenset(s)]),
                exact_shapley(players, lambda s: v_st[frozenset(s)]))

    raw, st = values(runs)
    rng = np.random.default_rng(seed)
    br, bs = {p: [] for p in players}, {p: [] for p in players}
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sub = pd.concat([runs[runs.seed == s] for s in pick])
        r, s_ = values(sub)
        for p in players:
            br[p].append(r[p])
            bs[p].append(s_[p])
    rows = []
    for p in players:
        rows.append({"region": p, "shapley_gain": raw[p], "gain_ci": tuple(np.percentile(br[p], [2.5, 97.5])),
                     "shapley_structure": st[p], "structure_ci": tuple(np.percentile(bs[p], [2.5, 97.5]))})
    return pd.DataFrame(rows).sort_values("shapley_structure", ascending=False)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(((p if not np.isnan(p) else 1.0), k) for k, p in pvals.items())
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def verdict_inferential(runs: pd.DataFrame, sesoi: float, alpha: float = 0.05) -> Verdict:
    """Inferential version of the deck's verdict rule (used for the headline
    verdicts after the mean-based pre-registered rule failed the Phase B
    hollow control -- see REPORT.md):

    clause 1  R5 > R0..R4: one-sided paired t-tests across seeds, Holm-corrected.
    clause 2  knockout falls to R3: knockout < R5 (one-sided paired p < alpha)
              AND knockout is not more than `sesoi` above R3 (one-sided test of
              H0: KO - R3 >= sesoi rejected at alpha).
    regularizer  R5 not significantly above R3 AND R5 TOST-equivalent to R3
              within +/- sesoi.
    `sesoi` = smallest effect size of interest, fixed per phase from controls
    (Phase A: the pre-registered 0.05; Phase B: 0.02 ~ half the planted
    brain's structure-specific gain)."""
    means = runs.groupby("rung")["score"].mean()
    p_raw = {r: p_greater(*paired(runs, "R5", r)) for r in LADDER[:-1]}
    p_adj = holm(p_raw)
    clause1 = all(p_adj[r] < alpha for r in p_adj)
    x5, xk = paired(runs, "R5", "knockout")
    k, r3 = paired(runs, "knockout", "R3")
    clause2 = False
    p_ko_below = p_greater(x5, xk) if len(x5) > 1 else np.nan
    p_ko_near = np.nan
    if len(k) > 1:
        d = k - r3
        se = d.std(ddof=1) / np.sqrt(len(d)) + 1e-12
        p_ko_near = float(stats.t.cdf((d.mean() - sesoi) / se, len(d) - 1))  # H0: diff >= sesoi
        clause2 = bool(p_ko_below < alpha and p_ko_near < alpha)
    x5b, x3 = paired(runs, "R5", "R3")
    t53 = tost_equivalence(x5b, x3, margin=sesoi) if len(x5b) > 1 else None
    reg = bool(t53 is not None and t53.equivalent and not (p_raw["R3"] < alpha))
    if clause1 and clause2:
        label = "GENUINE TRANSFER"
    elif reg:
        label = "REGULARIZER"
    elif clause1:
        label = "TRANSFER (knockout incomplete)"
    else:
        label = "INCONCLUSIVE"
    det = {"means": means.to_dict(), "p_holm_R5_gt": p_adj, "p_raw_R5_gt": p_raw, "p_ko_below_R5": p_ko_below,
           "p_ko_within_sesoi_of_R3": p_ko_near, "tost_r5_r3": (t53.mean_diff, t53.p_tost) if t53 else None,
           "sesoi": sesoi, "d_R5_R4": cohens_d(*paired(runs, "R5", "R4"))}
    return Verdict(label, clause1, clause2, reg, False, det)
