"""Phase B scaffold — NOT implemented yet.

Students need a reasoning task matching the fMRI paradigm: syllogisms,
zero-shot to more premises / propositional logic. Should mirror
nullladder.sandbox.task's shape (inputs, labels, probe_mask) so the same
training loop / zero-shot evaluation code in cdb2m/training and cdb2m/eval
works unmodified once this is implemented.
"""
from __future__ import annotations


class SyllogismTask:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase B scaffold: implement after Phase A passes")
