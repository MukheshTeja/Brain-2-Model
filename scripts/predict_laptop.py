"""Predict from a saved small model; no brain data or logic solver at inference.

Input JSON: {"premises": [["MORE",0,1],["MORE",1,2]],
             "conclusion": ["MORE",0,2]}
Entity IDs are integers 0..9. Supported statement types are listed by --help.
"""
import argparse
import json
from pathlib import Path
import pandas  # host Windows native-library load order
import torch
from cdb2m.fmri.reasoning_task import Problem, problems_to_batch, STMT_TYPES, N_SLOTS
from cdb2m.training.laptop import make_model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--input",required=True,help=f"JSON file; statement types: {STMT_TYPES}")
    args=ap.parse_args();torch.set_num_threads(1)
    raw=json.loads(Path(args.input).read_text())
    if not raw.get("premises"):
        ap.error("At least one premise is required")
    for s in raw["premises"]+[raw["conclusion"]]:
        if len(s)!=3 or s[0] not in STMT_TYPES or any(type(v) is not int or not 0<=v<N_SLOTS for v in s[1:]):
            ap.error(f"Invalid statement: {s}")
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model=make_model(ckpt["model_kind"],0).eval();model.load_state_dict(ckpt["model_state"])
    p=Problem("inference",[tuple(s) for s in raw["premises"]],tuple(raw["conclusion"]),0,0,False)
    batch=problems_to_batch([p])
    with torch.inference_mode():
        probabilities=model(batch["inputs"])[0][batch["probe_mask"]].softmax(-1)[0]
    print(json.dumps({"prediction":"valid" if probabilities.argmax().item() else "invalid",
                      "probability_valid":float(probabilities[1]),"probabilities_are_uncalibrated":True}))


if __name__=="__main__":
    main()
