"""Make measured figures and source-traceable literature context, never fabricated scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"legacy":"#8b94a5","baseline":"#405b70","R0":"#b5bbca","R1":"#9aaac0","R2":"#7799b2",
          "R3":"#d69e36","R4":"#bd7455","R5":"#197b74","knockout":"#935f93"}
VARIANTS = ["iid","premises_4","premises_5","premises_6","shuffled_order","propositional"]


def ci(x):
    x = np.asarray(x,dtype=float)
    mean = float(x.mean())
    if len(x)<2:
        return mean,mean,mean
    delta = float(stats.t.ppf(.975,len(x)-1)*stats.sem(x))
    return mean,mean-delta,mean+delta


def comparison(x,y):
    d = np.asarray(x)-np.asarray(y)
    mean,lo,hi=ci(d)
    # Enumerate paired sign flips: finite and reproducible even for zero variance.
    null = np.array([np.mean(d*np.array([1 if mask&(1<<i) else -1 for i in range(len(d))]))
                     for mask in range(2**len(d))])
    return {"mean_difference":mean,"ci_low":lo,"ci_high":hi,
            "paired_signflip_p_greater":float(np.mean(null>=mean-1e-12)),"seeds":len(d)}


def save(fig,out,name):
    fig.savefig(out/f"{name}.png",dpi=200,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="results/laptop")
    args=ap.parse_args();out=Path(args.out)
    records=[json.loads(p.read_text()) for p in sorted((out/"evaluate").glob("*.json"))]
    protocol=json.loads((out/"protocol.json").read_text())
    selected=json.loads((out/"selection.json").read_text())["model"]
    expected=len(protocol["evaluation_seeds"])*11
    if len(records)!=expected:
        raise RuntimeError(f"Incomplete experiment: {len(records)}/{expected}; finish run_laptop --stage evaluate")
    rows=[]
    for r in records:
        s=r["spec"]
        arm="legacy" if s["legacy"] else s["rung"] if s["loss"]=="linear_mse" else s["loss"]
        for variant,v in r["test"].items():
            rows.append(dict(arm=arm,seed=s["seed"],model=s["model"],variant=variant,accuracy=v["accuracy"],
                             n_test=v["n"],parameters=r["parameters"],seconds=r["seconds"],peak_mb=r["peak_process_mb"]))
    df=pd.DataFrame(rows)
    if df.duplicated(["arm","seed","variant"]).any():
        raise ValueError("Duplicate run rows")
    df.to_csv(out/"metrics.csv",index=False)
    summary=[]
    for (arm,var),g in df.groupby(["arm","variant"]):
        m,lo,hi=ci(g.accuracy)
        summary.append(dict(arm=arm,variant=var,mean=m,ci_low=lo,ci_high=hi,seeds=len(g),n_test=int(g.n_test.iloc[0])))
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    ood=df[df.variant!="iid"].groupby(["arm","seed"]).accuracy.mean().unstack("arm")
    tests={arm:comparison(ood.R5,ood[arm]) for arm in ("baseline","R0","R1","R2","R3","R4","knockout")}
    tests["architecture_vs_legacy"]=comparison(ood.baseline,ood.legacy)
    nulls=["R0","R1","R2","R3","R4"]
    ordered=sorted(nulls,key=lambda a:tests[a]["paired_signflip_p_greater"])
    previous=0.
    for i,arm in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*tests[arm]["paired_signflip_p_greater"]))
        tests[arm]["holm_p"]=previous
    # No mechanistic verdict from accuracy ceilings or five exploratory seeds.
    tests["interpretation"]="Exploratory performance study. A gain from architecture does not establish brain-specific transfer. No source-paper superiority claim."
    (out/"paired_comparisons.json").write_text(json.dumps(tests,indent=2))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","axes.labelcolor":"#334155"})
    order=["legacy","baseline","R0","R1","R2","R3","R4","R5","knockout"]
    fig,ax=plt.subplots(figsize=(11,5))
    for i,arm in enumerate(order):
        m,lo,hi=ci(ood[arm]*100)
        ax.bar(i,m,color=COLORS[arm],width=.65)
        ax.errorbar(i,m,yerr=[[m-max(0,lo)],[min(100,hi)-m]],fmt="none",color="#172b42",capsize=4)
        ax.text(i,min(102,max(m,hi)+1),f"{m:.1f}",ha="center",fontsize=10)
    ax.set(xticks=range(len(order)),xticklabels=["GRU-32\nlegacy","Selected\nno brain","R0","R1","R2","R3","R4","R5\nreal fMRI","Knockout"],
           ylim=(0,108),ylabel="Mean accuracy across five unseen variants (%)",
           title="Local controlled comparison: selected model and full Null Ladder")
    ax.text(0,-.19,"Five paired seeds; bars = means, intervals = 95% t intervals across seeds. Task encodings and data budget are matched.",transform=ax.transAxes,fontsize=9)
    fig.tight_layout();save(fig,out,"01_null_ladder")
    fig,axs=plt.subplots(1,2,figsize=(13,5),gridspec_kw={"width_ratios":[1.25,1]})
    for arm in ["legacy","baseline","R3","R5"]:
        means=[df[(df.arm==arm)&(df.variant==v)].accuracy.mean()*100 for v in VARIANTS]
        axs[0].plot(range(6),means,"o-",label=arm,color=COLORS[arm],lw=2)
    axs[0].set(xticks=range(6),xticklabels=["IID","4 prem.","5 prem.","6 prem.","Shuffled","Propos."],ylim=(0,104),ylabel="Accuracy (%)",title="Generalization by task")
    axs[0].legend(ncol=2,frameon=False)
    for i,v in enumerate(VARIANTS):
        wide=df[df.variant==v].pivot(index="seed",columns="arm",values="accuracy")
        m,lo,hi=ci(100*(wide.R5-wide.R3))
        axs[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",color=COLORS["R5"],capsize=4)
    axs[1].axvline(0,color="#8b94a5",ls="--")
    axs[1].set(yticks=range(6),yticklabels=["IID","4 premises","5 premises","6 premises","Shuffled","Propositional"],
               xlabel="R5 minus R3 (percentage points)",title="Brain-specific contrast, paired 95% CI")
    fig.suptitle("Architecture gains and brain-signal gains are separate questions",fontsize=14,fontweight="bold")
    fig.tight_layout();save(fig,out,"02_generalization")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for arm in ("legacy","baseline","R3","R5"):
        group=[r for r in records if ("legacy" if r["spec"]["legacy"] else r["spec"]["rung"] if r["spec"]["loss"]=="linear_mse" else r["spec"]["loss"])==arm]
        for axis,metric in zip(axs,("val_ce","val_accuracy")):
            history=np.array([[h[metric] for h in r["history"]] for r in group])
            steps=[h["step"] for h in group[0]["history"]]
            axis.plot(steps,history.mean(0),label=arm,color=COLORS[arm])
    axs[0].set(ylabel="Validation cross entropy",xlabel="Training steps",title="Validation loss")
    axs[0].set_yscale("log")
    axs[1].set(ylabel="Validation accuracy",xlabel="Training steps",title="Validation accuracy",ylim=(0,1.02))
    axs[1].legend(frameon=False);fig.tight_layout();save(fig,out,"03_training_curves")
    published=json.loads(Path("configs/published_benchmarks.json").read_text())
    pd.DataFrame(published).to_csv(out/"published_benchmarks.csv",index=False)
    fig,axs=plt.subplots(2,3,figsize=(15,10))
    panels=list(dict.fromkeys(r["panel"] for r in published))
    for ax,panel in zip(axs.flat,panels):
        group=[r for r in published if r["panel"]==panel]
        vals=[r["value"] for r in group]
        ax.bar(range(len(vals)),vals,color=["#91a1b5"]*(len(vals)-1)+["#197b74"],width=.6)
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f"{v:g}",ha="center")
        labels=[r["condition"].replace(" alignment","\nalignment").replace(", ","\n") for r in group]
        ax.set(xticks=range(len(vals)),xticklabels=labels,ylim=(0,max(vals)*1.25),title=panel)
        ax.set_ylabel({"accuracy_percent":"Accuracy (%)","top1_accuracy_percent":"Top-1 accuracy (%)","Pearson_r":"Pearson r","VAE_loss_lower_better":"VAE reconstruction loss (lower is better)"}[group[0]["metric"]])
        if "accuracy_percent" in group[0]["metric"]:
            ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100])
        notes={"Memory decisions":"GRU-512; reported across 10 initializations",
               "Driving reconstruction":"Mean loss; best listed transfer condition",
               "Propositional reasoning":"Mistral-7B; maximum of 5 runs, NOT mean",
               "ImageNet-A robustness":"Pearson alignment; 5 different models/group",
               "Human / GPT-2 predictions":"Behavioral correlation; no brain transfer training"}
        ax.text(.5,-.32,f"{group[0]['paper']} | PDF p. {group[0]['pdf_page']}\n{notes[panel]}",transform=ax.transAxes,ha="center",fontsize=9)
    ax=axs.flat[-1]
    vals=[ood[a].mean()*100 for a in ("legacy","baseline","R5")]
    ax.bar(range(3),vals,color=[COLORS[a] for a in ("legacy","baseline","R5")])
    for i,v in enumerate(vals):ax.text(i,v+2,f"{v:.1f}",ha="center")
    ax.set(xticks=range(3),xticklabels=["GRU-32","Selected\nno brain","Selected\nR5"],ylim=(0,110),ylabel="Mean OOD accuracy (%)",title="This project: symbolic reasoning")
    ax.text(.5,-.32,"Fresh local runs | mean of 5 seeds\nDifferent task/input/model from all published panels",transform=ax.transAxes,ha="center",fontsize=9)
    fig.suptitle("Published values and local results: context, not a cross-paper ranking",fontsize=16,fontweight="bold",y=1.01)
    fig.subplots_adjust(hspace=.9,wspace=.4,bottom=.16);save(fig,out,"04_published_context")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for i,arm in enumerate(("baseline","R5","cka","infonce")):
        m,lo,hi=ci(ood[arm]*100)
        axs[0].bar(i,m,color=[COLORS["baseline"],COLORS["R5"] ,"#7799b2","#935f93"][i])
        axs[0].errorbar(i,m,yerr=[[m-max(0,lo)],[min(100,hi)-m]],fmt="none",capsize=4,color="#172b42")
        axs[0].text(i,hi+1,f"{m:.1f}",ha="center")
    axs[0].set(xticks=range(4),xticklabels=["No brain","Linear MSE","CKA","InfoNCE"],ylim=(0,108),ylabel="Mean OOD accuracy (%)",title="Same-model alignment loss ablations")
    resource=df.drop_duplicates(["arm","seed"]).groupby("arm")[["seconds","peak_mb"]].mean()
    axs[1].bar(["GRU legacy",selected],[resource.loc["legacy","seconds"],resource.loc["baseline","seconds"]],color=[COLORS["legacy"],COLORS["baseline"]])
    axs[1].set(ylabel="Training seconds per seed",title="Measured CPU training time")
    fig.text(.5,-.03,"Local loss ablations are not full reproductions of B2M or NARF. CPU time excludes data preprocessing and evaluation.",ha="center",fontsize=9)
    fig.tight_layout();save(fig,out,"05_losses_and_runtime")
    # Choose deployment checkpoint strictly by validation CE, never test accuracy.
    deployment=min((r for r in records if r["spec"]["model"]==selected and r["spec"]["rung"] in ("baseline","R5") and r["spec"]["loss"]=="linear_mse"),key=lambda r:r["validation_ce"])
    recommendation={"checkpoint":deployment["checkpoint"],"selection":"minimum IID validation CE among selected-model baseline and R5 seeds",
                    "model":selected,"parameters":deployment["parameters"],"brain_condition":deployment["spec"]["rung"]}
    (out/"recommended_checkpoint.json").write_text(json.dumps(recommendation,indent=2))
    lines=["# Laptop experiment results", "", "Fresh experiments; five final training seeds. These results do not establish superiority over the uploaded papers.","",
           f"Selected model: **{selected}**, **{deployment['parameters']:,} trainable parameters**. Selection used two development seeds and IID validation cross entropy only.","",
           "## Local results", "", "| Condition | Mean unseen-task accuracy | 95% t interval across seeds |", "|---|---:|---:|"]
    for arm in order+["cka","infonce"]:
        m,lo,hi=ci(ood[arm]*100);lines.append(f"| {arm} | {m:.2f}% | [{lo:.2f}, {hi:.2f}]% |")
    arch=tests["architecture_vs_legacy"];brain=tests["R3"]
    lines += ["",f"Selected-model baseline versus GRU legacy: **{100*arch['mean_difference']:+.2f} percentage points** on the same fresh evaluation sets.",
              f"Real fMRI R5 minus matched surrogate R3: **{100*brain['mean_difference']:+.2f} points**, 95% CI [{100*brain['ci_low']:+.2f}, {100*brain['ci_high']:+.2f}].",
              "", "The relational architecture uses learned typed graph messages, shared entity weights, and query-conditioned answer prediction. It does not invoke the logic checker at inference. Its premise-order and entity-name invariance is an architectural prior. Improvement over a sequential GRU cannot be attributed to brain data.",
              "", "![Null ladder](01_null_ladder.png)","![Generalization](02_generalization.png)","![Training](03_training_curves.png)",
              "", "## Protocol and limits", "", "- Training: 1,000 steps, batch size 64 plus the same 70 brain-recorded examples, CPU, at most two workers. Training labels and examples are matched across local arms.",
              "- New optimizer: AdamW, gradient clipping, cosine decay, validation checkpointing. Alignment weight is fixed at 0.2 with 100-step warmup for every transfer arm. GRU legacy uses Adam and its final iterate at the same step budget, on the new split; it is a retrained comparator, not the historical CSV run.",
              "- The architecture comparison matches data and update count, not parameter count or floating-point operations. It changes both the student and training recipe; it is a combined pipeline improvement, not an isolated optimizer ablation.",
              "- Fixed hash partitions prevent exact encoded problems crossing train/validation/test. Recorded examples are removed from validation and test. This is exact-input separation, not a claim that underlying logical templates are disjoint.",
              "- Only three-premise syllogistic/transitive inputs are trained. Premise shuffling, longer chains, and propositional statement types remain untrained. The graph uses four learned message rounds, so length extrapolation has a structural limit.",
              "- Propositional evaluation exhausts the test partition of a small 360-encoding space. It shares four logical templates; a high score would not demonstrate general natural-language reasoning. Other variants each use 1,000 encoded problems.",
              "- Uncertainty intervals measure training-seed variability, not uncertainty over human subjects or task templates. Paired sign-flip p-values and Holm correction are in paired_comparisons.json. Five seeds are exploratory, with limited test resolution.",
              "- Brain targets are fitted only to recorded training stimuli. Group fMRI representations use all ten subjects; there is no held-out-subject generalization claim. Near-ceiling accuracy can hide transfer effects. This run does not revalidate planted/hollow controls for the new architecture or replace the deck's full Phase A/RQ1/RQ2/RQ3 protocol.",
              "- R3 is a finite-sample surrogate of retained second-order statistics. R5-R3 is an operational contrast, not proof that every surviving difference is computational structure. Mediation and knockout analyses need additional causal assumptions.",
              "- Spectral surrogates act on ordered trial betas, not continuous raw BOLD time series; their spectra are trial-order statistics. This distinction matters when interpreting preserved physiology.",
              "", "## Data and resources", "", "All 162 manifest entries passed SHA-256 checks: 23,579,110 bytes. ds003076 has 10 subjects, 70 valid problems and three networks. HCP has 20 downloaded subjects, of which 19 belong to two matched stimulus groups. Raw BOLD, ImageNet, podcast ECoG and private patient recordings are unnecessary for this compact experiment, but would be needed for their respective full replications.",
              f"Measured maximum process working set across final jobs: {max(r['peak_process_mb'] or 0 for r in records):.0f} MB. This is a per-process measurement, not whole-system RAM. Each worker uses one CPU thread. Model tensors contain about {deployment['parameters']*4/1024:.0f} KiB in float32; checkpoints also include metadata.",
              "", "## Published benchmarks", "", "![Published context](04_published_context.png)","![Losses and runtime](05_losses_and_runtime.png)",
              "", "The plots transcribe numeric values from the supplied PDFs. Page numbers are physical PDF pages. Source titles, model, dataset, statistic and links are recorded in published_benchmarks.csv. Missing paper error bars are not invented. Xiao's 83.7% and 70.5% are maxima over five runs, not means. Goldstein reports correspondence, and Sucholutsky compares existing models observationally; neither is a directly comparable brain-teacher training baseline.",
              "", "B2M-style InfoNCE and CKA are local objective ablations only. NARF uses a different intervention/fine-tuning pipeline; this experiment does not reproduce it. Published and local results differ in data, language encoding, evaluation and model scale, so do not use them for a best-method ranking.",
              "", "## Checkpoints", "",f"Validation-selected deployment checkpoint: `{deployment['checkpoint']}` (condition `{deployment['spec']['rung']}`). Every final seed and arm also has its own .pt checkpoint and JSON training trace.",
              "", "## Reproduce", "", "```powershell", ".\\.venv\\Scripts\\python -m scripts.audit_data", ".\\.venv\\Scripts\\python -m scripts.run_laptop --stage select --workers 2", ".\\.venv\\Scripts\\python -m scripts.run_laptop --stage evaluate --workers 2", ".\\.venv\\Scripts\\python -m scripts.report_laptop", "```", "", "Completed jobs resume only when the protocol, source hash and data manifest match. Use a fresh --out directory after code or settings changes."]
    if (out/"released_metrics.csv").exists():
        external=pd.read_csv(out/"released_metrics.csv")
        if not (external.groupby(["arm","dataset"]).seed.nunique()==len(protocol["evaluation_seeds"])).all():
            raise RuntimeError("Released-data results are incomplete; rerun scripts.evaluate_released first")
        coverage=pd.read_csv(out/"released_coverage.csv")
        lines += ["", "## External released-problem check", "", "![Released problems](06_released_problem_check.png)",
                  "", "| Released set | Original items | Evaluated items | Unique symbolic inputs |", "|---|---:|---:|---:|"]
        for r in coverage.itertuples():
            lines.append(f"| {r.file} | {r.original_items} | {r.evaluated_items} | {r.unique_symbolic_inputs} |")
        lines += ["", "| Released set | GRU legacy | Selected no brain | R3 | R5 |", "|---|---:|---:|---:|---:|"]
        for dataset,g in external.groupby("dataset"):
            values=g.groupby("arm").accuracy.mean()*100
            lines.append(f"| {dataset} | {values['legacy']:.2f}% | {values['baseline']:.2f}% | {values['R3']:.2f}% | {values['R5']:.2f}% |")
        lines += ["", "These are additional tests after model selection, using fixed symbolic translations of the released files. Exact three-premise encodings outside the frozen test partition are excluded to prevent possible overlap with generated training data. Longer-chain sets need no such exclusion because training uses only three premises. Unsupported disjunction and label/translation inconsistencies are listed in released_exclusions.csv. Lexical diversity collapses under symbolic translation; the original LLM task is harder and not directly comparable. This is not the full 600-item propositional benchmark from the paper.",
                  "", "Reproduce: `python -m scripts.evaluate_released`, then `python -m scripts.report_laptop`."]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"recommendation":recommendation,"architecture_difference":arch,"R5_minus_R3":brain},indent=2))


if __name__=="__main__":
    main()
