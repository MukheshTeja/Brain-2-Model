"""Turn results/phaseA*.csv + results/phaseB*.csv into figures, tables,
verdicts.json and REPORT.md.

    python -m scripts.make_report                 # full-preset results
    python -m scripts.make_report --preset quick  # quick-preset results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from cdb2m.analysis.core import (add_gains, fingerprint, mediation, paired, region_shapley,  # noqa: E402
                                 verdict, verdict_inferential, zero_shot)

SESOI = {"A": 0.05, "B": 0.02}  # smallest effect of interest per phase (A: pre-registered margin; B: ~half the planted structure gain)

RUNG_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "knockout", "KO_oracle"]
RUNG_LABEL = {"R0": "R0\nGaussian", "R1": "R1\nphase", "R2": "R2\ncov", "R3": "R3\nmax-ent", "R4": "R4\nshuffled",
              "R5": "R5\nreal", "knockout": "KO\nLEACE", "KO_oracle": "KO\noracle"}
# categorical slots 1-3 of the reference palette (validated all-pairs, light surface)
C = {"structured": "#2a78d6", "planted": "#2a78d6", "real_dr": "#2a78d6", "hcp": "#2a78d6",
     "hollow": "#eb6834", "planted_realsnr": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STUDENT_NAME = {"gru": "GRU", "transformer": "Transformer", "ppo": "PPO agent"}
ARM_NAME = {"structured": "structured brain", "hollow": "hollow brain", "planted": "planted (known structure)",
            "real_dr": "real fMRI (DR network)", "planted_realsnr": "planted @ real-fMRI SNR", "hcp": "real fMRI (HCP, MD)"}


def _style(ax, title=None):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def ladder_plot(zg: pd.DataFrame, arms, students, alpha, path, title):
    fig, axes = plt.subplots(1, len(students), figsize=(4.6 * len(students), 3.6), squeeze=False, sharey=True)
    for ax, st in zip(axes[0], students):
        sub = zg[(zg.student == st) & (zg.alpha == alpha)]
        for arm in arms:
            g = sub[sub.arm == arm]
            rungs = [r for r in RUNG_ORDER if r in set(g.rung)]
            if not rungs:
                continue
            m = g.groupby("rung")["score"].mean().reindex(rungs)
            se = g.groupby("rung")["score"].sem().reindex(rungs)
            x = [RUNG_ORDER.index(r) for r in rungs]
            ax.errorbar(x, m.values, yerr=se.values, color=C.get(arm, MUTED), marker="o", ms=5, lw=2, capsize=2,
                        label=ARM_NAME.get(arm, arm))
            ax.annotate(ARM_NAME.get(arm, arm), (x[-1], m.values[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=MUTED, va="center")
        ax.set_xticks(range(len(RUNG_ORDER)))
        ax.set_xticklabels([RUNG_LABEL[r] for r in RUNG_ORDER], fontsize=7)
        _style(ax, STUDENT_NAME.get(st, st))
    axes[0][0].set_ylabel("zero-shot gain vs no transfer\n(mean over unseen variants ± SEM)", fontsize=8, color=MUTED)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def dose_plot(zg: pd.DataFrame, panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4), squeeze=False, sharey=False)
    cols = {"R5": "#2a78d6", "R3": "#eb6834", "R0": "#1baf7a"}
    for ax, (phase, arm) in zip(axes[0], panels):
        sub = zg[(zg.phase == phase) & (zg.student == "gru") & (zg.arm == arm)]
        for rung in ("R0", "R3", "R5"):
            g = sub[sub.rung == rung].groupby("alpha")["score"]
            if g.ngroups == 0:
                continue
            m, se = g.mean(), g.sem()
            xs = [0.0] + list(m.index)
            ys = [0.0] + list(m.values)
            es = [0.0] + list(se.values)
            ax.errorbar(xs, ys, yerr=es, color=cols[rung], marker="o", ms=5, lw=2, capsize=2, label=rung)
        _style(ax, f"Phase {phase}: {ARM_NAME.get(arm, arm)}")
        ax.set_xlabel("transfer strength α", fontsize=8, color=MUTED)
    axes[0][0].set_ylabel("zero-shot gain (GRU)", fontsize=8, color=MUTED)
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def shapley_plot(tables: dict[str, pd.DataFrame], path):
    fig, axes = plt.subplots(1, len(tables), figsize=(4.6 * len(tables), 3.2), squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items()):
        t = t.sort_values("shapley_structure")
        y = np.arange(len(t))
        lo = t.shapley_structure - t.structure_ci.map(lambda c: c[0])
        hi = t.structure_ci.map(lambda c: c[1]) - t.shapley_structure
        ax.barh(y, t.shapley_structure, color="#2a78d6", height=0.55, xerr=[lo, hi], ecolor=MUTED, capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(t.region, fontsize=8)
        _style(ax, name)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("Shapley value of structure-specific gain\n[gain(R5) − gain(R3)], 95% bootstrap CI", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fingerprint_plot(fps: dict[str, pd.DataFrame], path):
    fig, axes = plt.subplots(1, len(fps), figsize=(4.4 * len(fps), 3.2), squeeze=False)
    for ax, (name, f) in zip(axes[0], fps.items()):
        x = np.arange(len(f))
        w = 0.38
        ax.bar(x - w / 2, f["R3"], width=w, color="#eb6834", label="R3 (brain statistics only)")
        ax.bar(x + w / 2, f["R5"], width=w, color="#2a78d6", label="R5 (real / planted)")
        ax.set_xticks(x)
        ax.set_xticklabels(f.variant, fontsize=7, rotation=25, ha="right")
        _style(ax, name)
    axes[0][0].set_ylabel("gain vs no transfer", fontsize=8, color=MUTED)
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--results", default="results")
    ap.add_argument("--thresholds", default="configs/phaseA_thresholds.yaml")
    args = ap.parse_args()
    tag = "" if args.preset == "full" else f"_{args.preset}"
    res = Path(args.results)
    out = res / f"report{tag}"
    out.mkdir(parents=True, exist_ok=True)
    thr = dict(OmegaConf.to_container(OmegaConf.load(args.thresholds)))

    frames = [pd.read_csv(p) for p in (res / f"phaseA{tag}.csv", res / f"phaseB{tag}.csv") if p.exists()]
    df = add_gains(pd.concat(frames, ignore_index=True))
    zg = zero_shot(df, "gain")  # per run: mean gain over unseen variants
    zg.to_csv(out / "runs_zero_shot_gain.csv", index=False)
    meta_p = res / f"phaseB{tag}_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    alpha_main = 0.2
    V, lines = {}, []
    L = lines.append

    # ---------------------------------------------------------------- verdicts
    def slice_runs(phase, dataset, student, arm):
        s = zg[(zg.phase == phase) & (zg.dataset == dataset) & (zg.student == student) & (zg.alpha == alpha_main)
               & (zg.arm == arm)]
        return s

    vt = []
    for (phase, dataset, student, arm), _ in zg[zg.alpha == alpha_main].groupby(["phase", "dataset", "student", "arm"]):
        if "+" in arm or arm.startswith("R_") or arm in ("all", "planted_realsnr") or "rq2" in dataset:
            continue
        runs = slice_runs(phase, dataset, student, arm)
        if runs.rung.nunique() < 6:
            continue
        v = verdict(runs, thr)  # pre-registered, mean-based
        vi = verdict_inferential(runs, SESOI[phase])  # headline
        m = v.details["means"]
        x5, x3 = paired(runs, "R5", "R3")
        dd = x5 - x3
        se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
        vt.append({"phase": phase, "dataset": dataset, "student": student, "arm": arm, "n_seeds": runs.seed.nunique(),
                   **{r: m.get(r, np.nan) for r in RUNG_ORDER}, "d_R5_R4": v.details["d_R5_R4"],
                   "p_R5>R3": v.details["p_R5_gt"].get("R3"), "p_holm_max": max(vi.details["p_holm_R5_gt"].values()),
                   "R5-R3": dd.mean(), "R5-R3_ci_lo": dd.mean() - 1.96 * se, "R5-R3_ci_hi": dd.mean() + 1.96 * se,
                   "verdict": vi.label, "verdict_prereg": v.label})
        V[f"{phase}|{dataset}|{student}|{arm}"] = {
            "verdict": vi.label, "verdict_prereg": v.label,
            "inferential": {k: (list(x) if isinstance(x, tuple) else x) for k, x in vi.details.items()},
            "prereg": {k: (list(x) if isinstance(x, tuple) else x) for k, x in v.details.items()}}
    vt = pd.DataFrame(vt)
    vt.to_csv(out / "verdicts.csv", index=False)

    def vrow(phase, dataset, student, arm):
        r = vt[(vt.phase == phase) & (vt.dataset == dataset) & (vt.student == student) & (vt.arm == arm)]
        return r.iloc[0] if len(r) else None

    def ladder_table(rows):
        hdr = "| student | arm | " + " | ".join(r for r in RUNG_ORDER) + " | seeds | p(R5>R3) | KO closes gap | verdict | pre-reg rule |"
        sep = "|" + "---|" * (len(RUNG_ORDER) + 7)
        body = []
        for r in rows:
            if r is None:
                continue
            cells = " | ".join(fmt(r[k]) for k in RUNG_ORDER)
            p = r["p_R5>R3"]
            gap = r["R5"] - r["R3"]
            closed = "–" if not (gap > 0.01) else f"{(r['R5'] - r['knockout']) / gap:.0%}"
            if gap > 0.01 and not np.isnan(r.get("KO_oracle", np.nan)):
                closed += f" (oracle {(r['R5'] - r['KO_oracle']) / gap:.0%})"
            body.append(f"| {STUDENT_NAME.get(r.student, r.student)} | {ARM_NAME.get(r.arm, r.arm)} | {cells} | "
                        f"{r.n_seeds} | {'–' if p is None or np.isnan(p) else f'{p:.3f}'} | {closed} | **{r.verdict}** | {r.verdict_prereg} |")
        return "\n".join([hdr, sep] + body)

    students = [s for s in ("gru", "transformer", "ppo") if s in set(zg.student)]
    # ------------------------------------------------------------- figures
    if (zg.phase == "A").any():
        ladder_plot(zg[(zg.phase == "A") & (zg.dataset == "sandbox_main")], ["structured", "hollow"], students,
                    alpha_main, out / "fig1_phaseA_ladder.png", "Phase A sandbox — Null Ladder (α = 0.2)")
    if (zg.dataset == "ds003076").any():
        ladder_plot(zg[zg.dataset == "ds003076"], ["planted", "hollow", "real_dr"], students, alpha_main,
                    out / "fig2_phaseB_ladder.png", "Phase B ds003076 — Null Ladder (α = 0.2)")
    if (zg.dataset == "hcp").any():
        ladder_plot(zg[zg.dataset == "hcp"], ["hcp"], ["gru"], alpha_main, out / "fig5_hcp_ladder.png",
                    "Replication: HCP relational task")
    dose_panels = [p for p in [("A", "structured"), ("A", "hollow"), ("B", "planted"), ("B", "real_dr")]
                   if ((zg.phase == p[0]) & (zg.arm == p[1])).any()]
    if dose_panels:
        dose_plot(zg[zg.dataset.isin(["sandbox_main", "ds003076"])], dose_panels, out / "fig3_dose_response.png")

    # ------------------------------------------------------------- RQ1
    med = {}
    for phase, dataset, arm in [("A", "sandbox_main", "structured"), ("A", "sandbox_main", "hollow"),
                                ("B", "ds003076", "planted"), ("B", "ds003076", "real_dr")]:
        d = zg[(zg.phase == phase) & (zg.dataset == dataset) & (zg.student == "gru") & (zg.arm == arm)
               & (zg.rung.isin(["R5", "baseline"]))].rename(columns={"score": "gain"})
        r = mediation(d)
        if r:
            med[f"{phase}|{arm}"] = r
    # ------------------------------------------------------------- RQ2
    shap = {}
    rq2a = zg[(zg.dataset == "sandbox_rq2")]
    if len(rq2a):
        shap["Phase A sandbox regions (known answer)"] = region_shapley(rq2a.rename(columns={"score": "gain"}))
    rq2b = zg[(zg.dataset == "ds003076") & zg.arm.isin(["DR", "MD", "LANG", "DR+MD", "DR+LANG", "MD+LANG", "DR+MD+LANG"])]
    if len(rq2b):
        shap["Phase B real fMRI networks"] = region_shapley(rq2b.rename(columns={"score": "gain"}))
    if shap:
        shapley_plot(shap, out / "fig4_rq2_shapley.png")
        for k, t in shap.items():
            t.to_csv(out / f"shapley_{'A' if 'Phase A' in k else 'B'}.csv", index=False)
    # ------------------------------------------------------------- fingerprints
    fps = {}
    for name, phase, dataset, arm in [("Phase A structured (GRU)", "A", "sandbox_main", "structured"),
                                      ("Phase B planted (GRU)", "B", "ds003076", "planted"),
                                      ("Phase B real fMRI (GRU)", "B", "ds003076", "real_dr")]:
        g = df[(df.phase == phase) & (df.dataset == dataset) & (df.student == "gru") & (df.arm == arm)
               & (df.alpha == alpha_main)]
        if len(g):
            fps[name] = fingerprint(g)
    if fps:
        fingerprint_plot(fps, out / "fig6_fingerprint.png")

    # ------------------------------------------------------------- REPORT.md
    L("# CD-B2M results\n")
    L(f"Preset: `{args.preset}` · transfer strength α = {alpha_main} for the ladder · every score is a **zero-shot gain**: "
      "accuracy on unseen task variants minus the paired no-transfer (α = 0) run with the same seed, data stream and "
      "evaluation sets. Ladder cells are means across seeds.\n")
    L("**Verdict rule** (deck slide 5: genuine transfer only if R5 beats every rung on zero-shot tests *and* the knockout "
      "falls to R3), tested inferentially across seeds:\n")
    L("- **GENUINE TRANSFER**: R5 > each of R0–R4 (one-sided paired t-tests, Holm-corrected, α = 0.05) **and** the "
      "knockout is significantly below R5 **and** no more than one SESOI above R3 (one-sided test).")
    L("- **REGULARIZER**: R5 is not significantly above R3 **and** TOST-equivalent to R3 within ± SESOI.")
    L("- **INCONCLUSIVE**: neither; **TRANSFER (knockout incomplete)**: clause 1 only.")
    L(f"- SESOI (smallest effect of interest): Phase A {SESOI['A']} (= the pre-registered margin in "
      f"`configs/phaseA_thresholds.yaml`), Phase B {SESOI['B']} (≈ half the planted brain's structure-specific gain; "
      "fixed from the controls, not from real data).\n")
    L("The *pre-reg rule* column applies the scaffold's original mean-based thresholds (R5 mean above every rung, "
      f"d(R5−R4) > {thr['min_effect_size_r5_vs_r4']}, ±{thr['knockout_vs_r3_margin']} TOST margins). It passes the Phase A "
      "sandbox but **fails the Phase B sandbox check**: it flags the task-free *hollow* control as genuine transfer, "
      "because Phase B effects (~0.01–0.05) are far below its 0.05 margins and mean ordering alone is noise-driven. "
      "That failure is why the headline verdicts use the inferential rule; both are shown.\n")
    L("**An instrument is only trusted where its sandbox check passes** (structured/planted → GENUINE TRANSFER, "
      "hollow → not transfer).\n")

    # sandbox check
    def lab(phase, dataset, st, arm):
        r = vrow(phase, dataset, st, arm)
        return None if r is None else r.verdict

    L("## 1. Phase A — synthetic brain sandbox (known answer)\n")
    L("![Phase A ladder](fig1_phaseA_ladder.png)\n")
    L(ladder_table([vrow("A", "sandbox_main", s, a) for s in students for a in ("structured", "hollow")]) + "\n")
    checks = []
    for s in students:
        ok_s = lab("A", "sandbox_main", s, "structured") == "GENUINE TRANSFER"
        ok_h = lab("A", "sandbox_main", s, "hollow") not in ("GENUINE TRANSFER", "TRANSFER (knockout incomplete)", None)
        checks.append((s, ok_s, ok_h))
        L(f"- **{STUDENT_NAME[s]}**: structured flagged as transfer {'✅' if ok_s else '❌'} · hollow not flagged {'✅' if ok_h else '❌'}")
    L("")
    L("*KO closes gap* = (R5 − knockout) / (R5 − R3): the share of the brain-specific gain that erasing the task "
      "variables removes (100% = falls exactly to R3); the oracle erases the known generative subspace exactly. "
      "When even the oracle overshoots or undershoots R3, the knockout ≈ R3 clause is limited by how that student "
      "reacts to R3 itself, not by incomplete erasure.\n")

    L("## 2. Phase B — real fMRI (ds003076 deductive reasoning)\n")
    if meta.get("reliability"):
        rl = meta["reliability"]
        L("Inter-subject RDM reliability (leave-one-subject-out r) — how much problem-level structure the group "
          "teacher signal can carry at all:\n")
        L("| signal | syllogism | transitive |\n|---|---|---|")
        for a in ("real_dr", "planted", "hollow", "planted_realsnr"):
            if f"{a}|syllogism" in rl:
                L(f"| {ARM_NAME[a]} | {rl[f'{a}|syllogism']:.3f} | {rl[f'{a}|transitive']:.3f} |")
        for a in ("MD", "LANG"):
            if f"coal:{a}|syllogism" in rl:
                L(f"| real fMRI ({a} network) | {rl[f'coal:{a}|syllogism']:.3f} | {rl[f'coal:{a}|transitive']:.3f} |")
        if "hcp|list1" in rl:
            L(f"| HCP relational (MD), lists 1/2 | {rl['hcp|list1']:.3f} | {rl['hcp|list2']:.3f} |")
        L("")
    L("![Phase B ladder](fig2_phaseB_ladder.png)\n")
    L(ladder_table([vrow("B", "ds003076", s, a) for s in students for a in ("planted", "hollow", "real_dr")]) + "\n")
    L("Phase B sandbox check (planted → GENUINE TRANSFER, hollow → not transfer):\n")
    for s in students:
        ok_s = lab("B", "ds003076", s, "planted") == "GENUINE TRANSFER"
        ok_h = lab("B", "ds003076", s, "hollow") not in ("GENUINE TRANSFER", "TRANSFER (knockout incomplete)", None)
        L(f"- **{STUDENT_NAME[s]}**: planted flagged {'✅' if ok_s else '❌'} · hollow not flagged {'✅' if ok_h else '❌'}"
          + ("" if ok_s and ok_h else " → this student's real-data verdict is **not interpretable**"))
    L("")
    fu = zg[(zg.phase == "B") & (zg.student == "gru") & (zg.alpha == alpha_main) & zg.arm.isin(["real_dr", "hcp", "planted_realsnr"])]
    if fu.seed.max() >= 10:
        L("**Follow-up seed extension (GRU).** After the 10-seed run left real-data R5−R3 neither significant nor "
          "equivalent, 20 more seeds were run for the real-data ladders, the HCP ladder and the power arm only "
          "(declared here as a sequential extension; first-10-seed results shown for comparison):\n")
        L("| slice | seeds | R5−R3 [95% CI] | p(R5>R3) |\n|---|---|---|---|")
        for arm in ("real_dr", "planted_realsnr", "hcp"):
            for lim in (10, 10 ** 6):
                g = fu[(fu.arm == arm) & (fu.seed < lim)]
                a5, a3 = paired(g, "R5", "R3")
                if len(a5) < 2:
                    continue
                dd = a5 - a3
                se = dd.std(ddof=1) / np.sqrt(len(dd))
                from scipy import stats as _st
                pv = _st.ttest_rel(a5, a3, alternative="greater").pvalue
                L(f"| {ARM_NAME[arm]} | {len(dd)} | {dd.mean():+.4f} [{dd.mean() - 1.96 * se:+.4f}, {dd.mean() + 1.96 * se:+.4f}] | {pv:.4f} |")
        L("")
    pw = zg[(zg.arm == "planted_realsnr") & (zg.alpha == alpha_main)]
    if len(pw):
        x5, x3 = paired(pw, "R5", "R3")
        x0 = paired(pw, "R5", "R0")
        L(f"**Power check — planted structure at the real data's SNR** (GRU, {len(x5)} seeds): "
          f"R5−R3 = {np.mean(x5 - x3):+.3f} (d = {np.mean(x5 - x3) / (np.std(x5 - x3, ddof=1) + 1e-12):.2f}), "
          f"R5−R0 = {np.mean(x0[0] - x0[1]):+.3f}. Noise scale used: {meta.get('realsnr_noise_scale')}.\n")

    L("## 3. Generalization fingerprints (GRU, α = 0.2)\n")
    L("Regularizers lift every variant a little; real structure lifts the variants that share it. "
      "`R5−R3` isolates the structure-specific part per variant.\n")
    if fps:
        L("![fingerprints](fig6_fingerprint.png)\n")
    for name, f in fps.items():
        L(f"**{name}**\n")
        L("| variant | R0 | R3 | R4 | R5 | knockout | R5−R3 | p(R5>R3) |\n|---|---|---|---|---|---|---|---|")
        for _, r in f.iterrows():
            pv = r["p(R5>R3)"]
            L(f"| {r.variant} | {fmt(r.R0)} | {fmt(r.R3)} | {fmt(r.R4)} | {fmt(r.R5)} | {fmt(r.knockout)} | "
              f"{fmt(r['R5-R3'])} | {'–' if np.isnan(pv) else f'{pv:.3f}'} |")
        L("")

    L("## 4. RQ1 — does brain-similarity *cause* better generalization?\n")
    if dose_panels:
        L("![dose response](fig3_dose_response.png)\n")
    L("α is the randomised dose; mediator = CKA(student, real/structured brain embedding); covariate = CKA(student, "
      "R3 surrogate); outcome = zero-shot gain. ACME = indirect effect of the dose through brain-similarity "
      "*after* controlling for similarity to the statistics-matched null (seed-clustered bootstrap 95% CI). "
      "ρ* = residual correlation an unmeasured confounder would need for ACME to vanish.\n")
    L("| slice | runs | dose→CKA (a) | CKA→gain (b) [95% CI] | ACME [95% CI] | direct effect ADE [95% CI] | partial r (p) | ρ* |\n|---|---|---|---|---|---|---|---|")
    for k, r in med.items():
        L(f"| {k} | {r['n_runs']} | {r['a_dose_to_cka']:+.3f} | {r['b_cka_to_gain']:+.3f} [{r['b_ci'][0]:+.3f}, {r['b_ci'][1]:+.3f}] | "
          f"{r['ACME']:+.4f} [{r['ACME_ci'][0]:+.4f}, {r['ACME_ci'][1]:+.4f}] | {r['ADE']:+.3f} [{r['ADE_ci'][0]:+.3f}, {r['ADE_ci'][1]:+.3f}] | "
          f"{r['partial_r_cka_gain']:+.2f} ({r['partial_p']:.3f}) | {r['sensitivity_rho']:+.2f} |")
    L("")

    L("## 5. RQ2 — which brain regions drive the gain?\n")
    if shap:
        L("![shapley](fig4_rq2_shapley.png)\n")
    for k, t in shap.items():
        L(f"**{k}** — exact Shapley over all coalitions; *gain* = raw transfer gain, *structure* = gain(R5) − gain(R3)\n")
        L("| region | Shapley (gain) [95% CI] | Shapley (structure) [95% CI] |\n|---|---|---|")
        for _, r in t.iterrows():
            L(f"| {r.region} | {r.shapley_gain:+.4f} [{r.gain_ci[0]:+.4f}, {r.gain_ci[1]:+.4f}] | "
              f"{r.shapley_structure:+.4f} [{r.structure_ci[0]:+.4f}, {r.structure_ci[1]:+.4f}] |")
        L("")

    L("## 6. RQ3 — does it hold across architectures?\n")
    L("| arm | " + " | ".join(STUDENT_NAME[s] for s in students) + " |\n|---|" + "---|" * len(students))
    for phase, dataset, arm in [("A", "sandbox_main", "structured"), ("A", "sandbox_main", "hollow"),
                                ("B", "ds003076", "planted"), ("B", "ds003076", "hollow"), ("B", "ds003076", "real_dr")]:
        L(f"| Phase {phase}: {ARM_NAME[arm]} | " + " | ".join(str(lab(phase, dataset, s, arm) or "–") for s in students) + " |")
    L("")
    if (zg.dataset == "hcp").any():
        L("## 7. Replication — HCP relational processing (MD network)\n")
        L("![hcp](fig5_hcp_ladder.png)\n")
        L(ladder_table([vrow("B", "hcp", "gru", "hcp")]) + "\n")

    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "verdicts.json").write_text(json.dumps({"verdicts": V, "mediation": med,
                                                   "shapley": {k: t.to_dict("records") for k, t in shap.items()}},
                                                  indent=1, default=str))
    print(f"wrote {out / 'REPORT.md'}")
    print(vt[["phase", "dataset", "student", "arm", "R3", "R4", "R5", "knockout", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
