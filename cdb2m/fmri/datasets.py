"""Phase B scaffold — NOT implemented yet (build after Phase A passes).

Loaders for the two fMRI datasets:
  - OpenNeuro ds003076 (deductive reasoning fMRI) — main dataset
  - HCP relational-processing fMRI — replication

Both loaders must return subject-level BOLD timeseries already masked to
the region set in regions.py, plus an events dataframe with columns at
least {onset, duration, trial_type, block_id} so design_check.py can
determine trial- vs block-level structure before any R4-style shuffling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FMRISubjectData:
    bold: np.ndarray  # [T, n_regions]
    events: pd.DataFrame
    tr: float


def load_ds003076(subject_id: str, bids_root: str) -> FMRISubjectData:
    raise NotImplementedError("Phase B scaffold: implement after Phase A passes")


def load_hcp_relational(subject_id: str, data_root: str) -> FMRISubjectData:
    raise NotImplementedError("Phase B scaffold: implement after Phase A passes")
