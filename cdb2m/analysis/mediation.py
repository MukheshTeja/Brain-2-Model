"""RQ1: alpha is the randomised dose; does CKA-to-brain mediate zero-shot
gain after controlling for CKA-to-R3? Implemented in
cdb2m.analysis.core.mediation (product-of-coefficients ACME with a
seed-clustered bootstrap + Imai-Keele-Yamamoto sensitivity rho); this
module keeps the scaffold's entry point."""
from __future__ import annotations

import pandas as pd

from cdb2m.analysis.core import add_gains, mediation, zero_shot


def run_mediation_analysis(results: pd.DataFrame, brain: str, student: str = "gru") -> dict:
    zg = zero_shot(add_gains(results), "gain")
    d = zg[(zg.arm == brain) & (zg.student == student) & zg.rung.isin(["R5", "baseline"])]
    return mediation(d.rename(columns={"score": "gain"}))
