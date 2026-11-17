"""External sanity check on Xiao's released problems after model selection.

This evaluates a symbolic translation, not the original LLM language interface.
Unsupported disjunctions and label inconsistencies are explicitly reported.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cdb2m.fmri import reasoning_task as rt
from cdb2m.fmri.datasets import load_ds003076
from cdb2m.training.laptop import make_model, score, problem_key, partition
from scripts.report_laptop import ci,save


def parse_propositional(row,seed):
    rule=row["premise1"].rstrip(".")
    match=re.fullmatch(r"If (.+), then (.+)",rule)
    if not match:
        raise ValueError("unsupported disjunction; model has no OR input type")
    atoms=[match.group(1),match.group(2)]
    slots=np.random.default_rng(seed).permutation(rt.N_SLOTS)
    def statement(text):
        text=text.rstrip(".")
        negative=" is not " in text
        affirmative=text.replace(" is not "," is ")
        if affirmative not in atoms:
            raise ValueError("unrecognized proposition")
        slot=int(slots[atoms.index(affirmative)])
        return ("ASSERTNOT" if negative else "ASSERT",slot,slot)
    p=rt.Problem("propositional",[("IFTHEN",int(slots[0]),int(slots[1])),statement(row["premise2"])],
                 statement(row["conclusion"]),int(row["trial_type"].endswith("_true")),2,False)
    # Labels verify the translation; they never choose the input encoding.
    if rt.evaluate_problem(p)!=p.label:
        raise ValueError("label mismatch")
    return p


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="results/laptop");ap.add_argument("--data-dir",default="data")
    args=ap.parse_args();out=Path(args.out);torch.set_num_threads(1)
    protocol=json.loads((out/"protocol.json").read_text())
    expected=11*len(protocol["evaluation_seeds"])
    if len(list((out/"evaluate").glob("*.json"))) != expected:
        raise RuntimeError("Finish all final evaluation jobs before making the released-data comparison")
    datasets={};coverage=[];excluded=[]
    rec={problem_key(p) for p in load_ds003076(args.data_dir,networks=("deductive_reasoning",)).problems}
    for file in sorted((Path(args.data_dir)/"xiao_generated_sets").glob("*.json")):
        raw=json.loads(file.read_text());probs=[];total=0
        for family,items in raw.items():
            for i,row in enumerate(items):
                total+=1
                try:
                    seed=92_000_000+i
                    if family=="propositional":
                        p=parse_propositional(row,seed)
                    else:
                        premises=[row[k] for k in sorted(row) if re.fullmatch(r"premise\d+",k)]
                        p=rt.parse_fmri_problem(premises,row["conclusion"],row["trial_type"],seed)
                    if problem_key(p) in rec:
                        raise ValueError("exact recorded-training encoding")
                    if len(p.premises)==3 and partition(p)!="test":
                        raise ValueError("excluded by fixed exact-input test partition")
                    probs.append(p)
                except (ValueError,IndexError) as exc:
                    excluded.append(dict(file=file.name,family=family,row=i,reason=str(exc)))
        datasets[file.stem]=probs
        coverage.append(dict(file=file.name,original_items=total,evaluated_items=len(probs),
                             unique_symbolic_inputs=len({problem_key(p) for p in probs}),excluded_items=total-len(probs)))
    pd.DataFrame(coverage).to_csv(out/"released_coverage.csv",index=False)
    pd.DataFrame(excluded).to_csv(out/"released_exclusions.csv",index=False)
    rows=[]
    for file in sorted((out/"evaluate").glob("*.json")):
        run=json.loads(file.read_text());spec=run["spec"]
        if not spec["legacy"] and (spec["rung"] not in ("baseline","R3","R5") or spec["loss"]!="linear_mse"):
            continue
        arm="legacy" if spec["legacy"] else spec["rung"]
        model=make_model(spec["model"],0).eval()
        model.load_state_dict(torch.load(run["checkpoint"],map_location="cpu",weights_only=True)["model_state"])
        for name,probs in datasets.items():
            correct=0.
            for start in range(0,len(probs),128):
                subset=probs[start:start+128]
                correct+=score(model,rt.problems_to_batch(subset))[0]*len(subset)
            rows.append(dict(arm=arm,seed=spec["seed"],dataset=name,accuracy=correct/len(probs),n=len(probs)))
    df=pd.DataFrame(rows);df.to_csv(out/"released_metrics.csv",index=False)
    fig,ax=plt.subplots(figsize=(12,5))
    order=list(datasets)
    for arm,color in [("legacy","#8b94a5"),("baseline","#405b70"),("R3","#d69e36"),("R5","#197b74")]:
        vals=[ci(df[(df.arm==arm)&(df.dataset==name)].accuracy*100) for name in order]
        ax.errorbar(range(len(order)),[v[0] for v in vals],yerr=[[v[0]-v[1] for v in vals],[v[2]-v[0] for v in vals]],fmt="o-",label=arm,color=color,capsize=3)
    labels=[name.replace("deductive_reasoning_data_test_","").replace("premises"," premises").replace("deductive_reasoning_data_test","3 premises\n(test partition)") for name in order]
    ax.spines[["top","right"]].set_visible(False)
    ax.set(xticks=range(len(order)),xticklabels=labels,ylabel="Accuracy (%)",ylim=(0,108),title="External check: symbolic translations of released Xiao problems")
    ax.legend(frameon=False,ncol=4)
    ax.text(0,-.22,"Five seeds; 95% t intervals. Subsets/translation differ from the paper's language evaluation.\nPropositional disjunction is unsupported and explicitly excluded; see coverage and exclusions CSVs.",transform=ax.transAxes,fontsize=10)
    fig.tight_layout();save(fig,out,"06_released_problem_check")
    print(pd.DataFrame(coverage).to_string(index=False))
    print(df.groupby(["arm","dataset"]).accuracy.mean().to_string())


if __name__=="__main__":
    main()
