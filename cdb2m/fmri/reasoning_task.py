"""Deductive-reasoning task matched to the ds003076 fMRI paradigm (Phase B).

A problem is a short sequence of STATEMENTS, one per timestep (premises,
then the conclusion), and the student must say valid (1) / invalid (0) at
the conclusion step. Each timestep is a factored one-hot:
    [statement type | subject slot | object slot | is-conclusion flag]
The batch layout (inputs [B, T, D], labels [B, T], probe_mask [B, T]) is the
same as the Phase A sandbox task, so the training loop, PPO agent and
zero-shot evaluation are shared.

(One-statement-per-step rather than one-word-per-step is a deliberate
compute choice: word-level variable binding did not rise above chance for
32-128-unit students within several thousand CPU steps, while the
statement-level version is learnt in seconds. The logic is identical.)

Problem families, exactly as in ds003076 (Lytle, Prado & Booth 2020):

* Syllogistic / set inclusion: "All A are B; All B are C; x is A |- x is C".
  Chain of m inclusion premises C0 < ... < Cm plus one fact about an
  individual x ("x is Ci" or "x is not Cj"). Affirmative fact x in Ci
  entails x in Ck for k >= i (modus ponens chain); negated fact x notin Cj
  entails x notin Ck for k <= j (modus tollens chain). Everything else --
  including the fallacies in the fMRI set (affirming the consequent,
  denying the antecedent) -- is invalid.
* Transitive / linear order: "P0 > P1; P1 > P2; P2 > P3 |- P1 > P3".
  "Pa > Pb" is valid iff a < b; "Pa is not > Pb" iff a > b.

"2_" / "3_" in the fMRI condition names = number of premises needed; here
the chain distance (+1 for the fact premise in syllogisms).

Entities (pseudowords, adjectives, names) are mapped to randomly assigned
SLOTS per problem, so the task requires binding variables, not memorising
words -- the reason the fMRI study used pseudowords.

Zero-shot suite (never trained on):
  premises_4/5/6  longer chains (more premises), as in Xiao et al. 2026
  shuffled_order  3-premise problems with the premise order permuted
  propositional   modus ponens / tollens + their fallacies, written with
                  IF-THEN / ASSERT statement types never seen in training
                  (far transfer: shares logic but no surface form)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

N_SLOTS = 10
STMT_TYPES = ["ALL", "IS", "ISNOT", "MORE", "NOTMORE", "IFTHEN", "ASSERT", "ASSERTNOT"]
TYPE = {t: i for i, t in enumerate(STMT_TYPES)}
INPUT_DIM = len(STMT_TYPES) + 2 * N_SLOTS + 1
N_CLASSES = 2

Stmt = tuple[str, int, int]  # (type, subject slot, object slot)


@dataclass
class Problem:
    family: str  # "syllogism" | "transitive" | "propositional"
    premises: list[Stmt]
    conclusion: Stmt
    label: int  # 1 valid / 0 invalid
    n_required: int
    negated: bool


# ----------------------------------------------------------------- generators
def _syllogism(rng, m, slots, distance, affirm_fact, negate_concl, valid) -> Problem:
    classes, x = slots[: m + 1], slots[m + 1]
    premises = [("ALL", classes[i], classes[i + 1]) for i in range(m)]
    cands = []
    for i in range(m + 1):
        for k in range(m + 1):
            if abs(k - i) != distance:
                continue
            v = ((not negate_concl) and k >= i) if affirm_fact else (negate_concl and k <= i)
            if v == valid:
                cands.append((i, k))
    i, k = cands[int(rng.integers(len(cands)))]
    fact = ("IS" if affirm_fact else "ISNOT", x, classes[i])
    concl = ("ISNOT" if negate_concl else "IS", x, classes[k])
    return Problem("syllogism", premises + [fact], concl, int(valid), distance + 1, negate_concl)


def _transitive(rng, m, slots, distance, negate_concl, valid) -> Problem:
    people = slots[: m + 1]
    premises = [("MORE", people[i], people[i + 1]) for i in range(m)]
    cands = [(a, b) for a in range(m + 1) for b in range(m + 1)
             if abs(a - b) == distance and ((a > b) if negate_concl else (a < b)) == valid]
    a, b = cands[int(rng.integers(len(cands)))]
    return Problem("transitive", premises, ("NOTMORE" if negate_concl else "MORE", people[a], people[b]),
                   int(valid), distance, negate_concl)


def _propositional(slots, kind) -> Problem:
    p, q = slots[0], slots[1]
    rule = ("IFTHEN", p, q)
    table = {
        "MP": ([rule, ("ASSERT", p, p)], ("ASSERT", q, q), 1, False),
        "AC": ([rule, ("ASSERT", q, q)], ("ASSERT", p, p), 0, False),  # affirming the consequent
        "MT": ([rule, ("ASSERTNOT", q, q)], ("ASSERTNOT", p, p), 1, True),
        "DA": ([rule, ("ASSERTNOT", p, p)], ("ASSERTNOT", q, q), 0, True),  # denying the antecedent
    }
    prem, concl, label, neg = table[kind]
    return Problem("propositional", prem, concl, label, 2, neg)


def generate_problems(n: int, seed: int, n_premises: int = 3,
                      families: tuple[str, ...] = ("syllogism", "transitive")) -> list[Problem]:
    """Balanced over family x validity x conclusion polarity x distance."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        fam = families[j % len(families)]
        slots = [int(s) for s in rng.permutation(N_SLOTS)]
        valid, neg = bool(rng.integers(2)), bool(rng.integers(2))
        if fam == "syllogism":
            m = n_premises - 1
            dist = int(rng.integers(1, m + 1))
            # a valid conclusion needs a fact of the same polarity; invalid ones use
            # either polarity (covers both fallacies seen in the fMRI set)
            affirm_fact = (not neg) if valid else bool(rng.integers(2))
            out.append(_syllogism(rng, m, slots, dist, affirm_fact, neg, valid))
        elif fam == "transitive":
            m = n_premises
            lo = 2 if n_premises == 3 else 1  # the fMRI set uses distances 2 and 3
            out.append(_transitive(rng, m, slots, int(rng.integers(lo, m + 1)), neg, valid))
        else:
            out.append(_propositional(slots, ["MP", "AC", "MT", "DA"][j % 4]))
    return out


def problems_to_batch(problems: list[Problem], seed: int = 0, shuffle_premises: bool = False) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    T = max(len(p.premises) for p in problems) + 1
    B = len(problems)
    X = np.zeros((B, T, INPUT_DIM), dtype=np.float32)
    y = np.zeros((B, T), dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    ans = np.zeros(B, dtype=np.int64)
    nt = len(STMT_TYPES)
    for b, p in enumerate(problems):
        order = rng.permutation(len(p.premises)) if shuffle_premises else range(len(p.premises))
        stmts = [p.premises[i] for i in order] + [p.conclusion]
        for t, (ty, a, c) in enumerate(stmts):
            X[b, t, TYPE[ty]] = 1.0
            X[b, t, nt + a] = 1.0
            X[b, t, nt + N_SLOTS + c] = 1.0
        t = len(stmts) - 1
        X[b, t, -1] = 1.0
        y[b, t], mask[b, t], ans[b] = p.label, True, t
    return {"inputs": torch.from_numpy(X), "labels": torch.from_numpy(y),
            "probe_mask": torch.from_numpy(mask), "answer_idx": torch.from_numpy(ans)}


class ReasoningEvalTask:
    """Adapter so cdb2m.eval.zero_shot.eval_accuracy can call .generate()."""

    def __init__(self, n_premises: int = 3, families=("syllogism", "transitive"), shuffle_premises: bool = False):
        self.n_premises, self.families, self.shuffle = n_premises, tuple(families), shuffle_premises
        self.input_dim, self.n_classes = INPUT_DIM, N_CLASSES

    def generate(self, batch_size: int, seed: int) -> dict[str, torch.Tensor]:
        probs = generate_problems(batch_size, 1_000_003 + seed + 7919 * self.n_premises, self.n_premises, self.families)
        return problems_to_batch(probs, seed=seed, shuffle_premises=self.shuffle)


def make_reasoning_zero_shot_suite() -> dict[str, ReasoningEvalTask]:
    return {
        "premises_4": ReasoningEvalTask(4),
        "premises_5": ReasoningEvalTask(5),
        "premises_6": ReasoningEvalTask(6),
        "shuffled_order": ReasoningEvalTask(3, shuffle_premises=True),
        "propositional": ReasoningEvalTask(2, families=("propositional",)),
    }


# --------------------------------------------- parse the real fMRI problems
_COMP = re.compile(r"^(\w+) is (not )?(\w+er) than (\w+)$")


def _norm(w: str) -> str:
    w = w.strip().lower()
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def parse_fmri_problem(premises: list[str], conclusion: str, trial_type: str, seed: int = 0) -> Problem:
    """Parse one ds003076 problem (English) into slot statements, then CHECK
    that the independent logic engine reproduces the dataset's own label."""
    sents = [" ".join(s.strip().rstrip(".").split()) for s in premises + [conclusion]]
    perm = [int(s) for s in np.random.default_rng(seed).permutation(N_SLOTS)]
    ents: dict[str, int] = {}

    def slot(w: str) -> int:
        w = _norm(w)
        if w not in ents:
            ents[w] = perm[len(ents)]
        return ents[w]

    def enc(s: str) -> Stmt:
        m = _COMP.match(s)
        if m:
            return ("NOTMORE" if m.group(2) else "MORE", slot(m.group(1)), slot(m.group(4)))
        w = s.split()
        if w[0].lower() == "all":  # "All trabs are fast" / "All fast things are brown"
            return ("ALL", slot(w[1]), slot(w[-1]))
        return ("ISNOT" if "not" in w else "IS", slot(w[0]), slot(w[-1]))  # "Abe is (not) (a) trab"

    prem = [enc(s) for s in sents[:-1]]
    concl = enc(sents[-1])
    family = "transitive" if any(p[0] == "MORE" for p in prem) else "syllogism"
    prob = Problem(family, prem, concl, 1 if "_true_" in trial_type else 0, int(trial_type[0]), "negate" in trial_type)
    if evaluate_problem(prob) != prob.label:
        raise ValueError(f"label mismatch for {sents} ({trial_type})")
    return prob


def evaluate_problem(p: Problem) -> int:
    """Independent forward-chaining logic engine (verifies parser + generator)."""
    if p.family == "propositional":
        rule = p.premises[0]
        fact = p.premises[1]
        ty, a, _ = p.conclusion
        if fact[0] == "ASSERT":
            return int(fact[1] == rule[1] and ty == "ASSERT" and a == rule[2])
        return int(fact[1] == rule[2] and ty == "ASSERTNOT" and a == rule[1])
    if p.family == "transitive":
        closure = {(q[1], q[2]) for q in p.premises}
        changed = True
        while changed:
            changed = False
            for (a, b) in list(closure):
                for (c, d) in list(closure):
                    if b == c and (a, d) not in closure:
                        closure.add((a, d))
                        changed = True
        ty, a, b = p.conclusion
        return int((b, a) in closure) if ty == "NOTMORE" else int((a, b) in closure)
    sub: dict[int, set[int]] = {}
    pos, negs = set(), set()
    for ty, a, b in p.premises:
        if ty == "ALL":
            sub.setdefault(a, set()).add(b)
        else:
            (negs if ty == "ISNOT" else pos).add(b)

    def up(c: int) -> set[int]:
        seen, st = {c}, [c]
        while st:
            for d in sub.get(st.pop(), ()):
                if d not in seen:
                    seen.add(d)
                    st.append(d)
        return seen

    known_in = set().union(*[up(c) for c in pos]) if pos else set()
    known_out = {c for c in range(N_SLOTS) if any(n in up(c) for n in negs)}
    ty, _, target = p.conclusion
    return int(target in known_out) if ty == "ISNOT" else int(target in known_in)
