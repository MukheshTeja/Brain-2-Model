"""RQ1 scaffold: alpha is the randomized dose; test whether CKA-to-brain
predicts zero-shot gain after controlling for CKA-to-R3 (mediation
analysis), plus a sensitivity analysis for the no-unmeasured-confounding
assumption.

Depends on cdb2m/eval/cka_report.py's per-rung CKA values (treat
CKA-to-"R5"/brain as the mediator candidate and CKA-to-"R3" as the
covariate to control for) joined against the sweep results' zero-shot
gain. Implement once Phase A's sweep has produced enough alpha levels to
treat alpha as a randomized dose with real variation in CKA.
"""
from __future__ import annotations

import pandas as pd


def run_mediation_analysis(results: pd.DataFrame, brain: str) -> dict:
    """Placeholder — not implemented in milestone 1.

    Planned approach: statsmodels mediation analysis
    (statsmodels.stats.mediation.Mediation) with alpha as treatment,
    CKA-to-brain as mediator, CKA-to-R3 as a covariate, zero-shot gain as
    outcome; report the average causal mediation effect (ACME) and a
    sensitivity parameter (rho) sweep per Imai et al. 2010.
    """
    raise NotImplementedError("RQ1 mediation analysis: scaffold only, implement after Phase A passes")
