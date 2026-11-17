"""Bounded CPU training with independent validation and portable checkpoints."""
from __future__ import annotations
import copy
import math
import time
import hashlib
import json
import torch
from torch import nn
from cdb2m.fmri import reasoning_task as rt
from cdb2m.students.gru import GRUStudent
from cdb2m.students.relational import RelationalStudent
from cdb2m.training.online import BlockAlignment


def problem_key(p):
    return json.dumps([p.premises, p.conclusion], separators=(",", ":"))


def partition(p):
    bucket = int.from_bytes(hashlib.blake2b(problem_key(p).encode(), digest_size=4).digest(), "little") % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def split_problems(n, seed, split, n_premises=3, families=("syllogism", "transitive"), excluded=()):
    out, seen = [], set(excluded)
    iteration = 0
    while len(out) < n:
        for p in rt.generate_problems(max(n, 64), seed + iteration*104729, n_premises, families):
            key = problem_key(p)
            if partition(p) == split and key not in seen:
                out.append(p); seen.add(key)
                if len(out) == n:
                    break
        iteration += 1
        if iteration > 1000:
            raise RuntimeError("Requested split is larger than the unique problem space")
    return out


def make_model(kind, seed):
    torch.manual_seed(seed)
    if kind == "relational32":
        return RelationalStudent(32)
    if kind in ("gru32", "gru64"):
        return GRUStudent(rt.INPUT_DIM, int(kind[3:]), 2)
    raise ValueError(kind)


@torch.no_grad()
def score(model, batch):
    model.eval()
    logits = model(batch["inputs"])[0][batch["probe_mask"]]
    labels = batch["labels"] [batch["probe_mask"]]
    return float((logits.argmax(-1) == labels).float().mean()), float(nn.functional.cross_entropy(logits, labels))


def fit(kind, seed, recorded, blocks=None, loss_name="linear_mse", alpha=.2,
        steps=1000, batch_size=64, lr=.003, legacy=False):
    model = make_model(kind, seed)
    alignment = BlockAlignment(loss_name, model.hidden_dim, blocks) if blocks else None
    params = list(model.parameters()) + (list(alignment.parameters()) if alignment else [])
    opt = torch.optim.Adam(params, lr=lr) if legacy else torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    # Validation uses a separate seed namespace, IID three-premise tasks only.
    # Recorded problems are brain-supervised training examples, so remove their
    # exact encodings even if the hash would otherwise put them in validation.
    def key_from_input(x):
        return x.numpy().tobytes()
    recorded_keys = {key_from_input(x) for x in recorded["inputs"]}
    val = rt.problems_to_batch(split_problems(768, 80_000_000, "validation"))
    keep = torch.tensor([key_from_input(x) not in recorded_keys for x in val["inputs"]])
    val = {k:v[keep] for k,v in val.items()}
    best, state, history = math.inf, None, []
    nrec = len(recorded["inputs"])
    start = time.perf_counter()
    for step in range(steps):
        model.train()
        fresh = rt.problems_to_batch(split_problems(batch_size, 10_000*seed + step, "train"))
        batch = {k: torch.cat([recorded[k], fresh[k]]) for k in recorded}
        logits, hidden = model(batch["inputs"])
        m = batch["probe_mask"]
        ce = nn.functional.cross_entropy(logits[m], batch["labels"][m])
        transfer = ce.new_zeros(())
        if alignment is not None:
            ans = hidden[torch.arange(nrec), recorded["answer_idx"]]
            transfer = alignment(ans)
        # Same objective across rungs; warmup avoids a random projection dominating early learning.
        a = alpha * min(1., (step+1)/100) if alignment is not None else 0.
        loss = (1-a)*ce + a*transfer
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss: {kind}, seed={seed}, step={step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not legacy:
            nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        if not legacy:
            for group in opt.param_groups:
                group["lr"] = lr * (.1 + .9*.5*(1+math.cos(math.pi*(step+1)/steps)))
        if (step+1) % 100 == 0 or step == steps-1:
            acc, vloss = score(model, val)
            history.append({"step": step+1, "train_ce": float(ce.detach()), "transfer_loss": float(transfer.detach()),
                            "val_accuracy": acc, "val_ce": vloss})
            if vloss < best:
                best, state = vloss, copy.deepcopy(model.state_dict())
    # Legacy comparator reproduces the old final-iterate policy. New arms use validation CE.
    if not legacy:
        model.load_state_dict(state)
    return model.eval(), {"history": history, "validation_ce": score(model,val)[1],
                          "validation_accuracy": score(model,val)[0], "seconds": time.perf_counter()-start,
                          "parameters": sum(p.numel() for p in model.parameters())}
