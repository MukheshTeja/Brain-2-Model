from dataclasses import replace
import numpy as np
import torch
from cdb2m.fmri import reasoning_task as rt
from cdb2m.students.relational import RelationalStudent
from cdb2m.training.laptop import split_problems, problem_key, partition, fit


def test_split_membership_and_exclusions():
    a = split_problems(50,123,"train")
    b = split_problems(50,123,"validation")
    c = split_problems(50,123,"test")
    assert all(partition(p)=="train" for p in a)
    assert not ({problem_key(p) for p in a} & {problem_key(p) for p in b+c})
    excluded = {problem_key(p) for p in c}
    d = split_problems(50,123,"test",excluded=excluded)
    assert not excluded & {problem_key(p) for p in d}


def test_relational_invariant_to_slot_names_and_premise_order():
    torch.manual_seed(1)
    model = RelationalStudent().eval()
    problems = rt.generate_problems(20,0)
    perm = np.random.default_rng(8).permutation(rt.N_SLOTS)
    def rename(s):
        return s[0],int(perm[s[1]]),int(perm[s[2]])
    renamed = [replace(p,premises=[rename(s) for s in p.premises],conclusion=rename(p.conclusion)) for p in problems]
    a = rt.problems_to_batch(problems)
    b = rt.problems_to_batch(renamed,seed=21,shuffle_premises=True)
    with torch.no_grad():
        assert torch.allclose(model(a["inputs"])[0][a["probe_mask"]],model(b["inputs"])[0][b["probe_mask"]],atol=1e-6)


def test_relational_checkpoint_and_gradient(tmp_path):
    model = RelationalStudent()
    batch = rt.problems_to_batch(rt.generate_problems(8,2,6))
    logits,hidden = model(batch["inputs"])
    loss = torch.nn.functional.cross_entropy(logits[batch["probe_mask"]],batch["labels"][batch["probe_mask"]])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path/'weights.pt'; torch.save(model.state_dict(),path)
    clone = RelationalStudent(); clone.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(logits,clone(batch["inputs"])[0])
    assert hidden.shape == (8,7,32)


def test_laptop_fit_finite_and_reproducible():
    rec = rt.problems_to_batch(rt.generate_problems(8,17))
    a,info = fit("gru32",2,rec,steps=2)
    b,_ = fit("gru32",2,rec,steps=2)
    assert np.isfinite(info["validation_ce"])
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
