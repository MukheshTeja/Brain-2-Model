"""Brain regions / networks used as RQ2's Shapley players.

Real-data path (default, what the pipeline runs): the three ROI NETWORKS for
which Xiao et al. 2026 released trial-wise betas -- deductive-reasoning
(DR), multiple-demand (MD), language (LANG). The released betas carry no
voxel coordinates, so finer anatomical regions cannot be separated from them.

Optional raw-BOLD path (`scripts/download_data.py --with-raw-bold`): the DR
network is the union of spheres around the Prado, Chadha & Booth (2011)
deductive-reasoning meta-analysis peaks below (Talairach, exactly as used
in Xiao et al.'s matlab/run_get_dr_roi.m), converted to MNI with the
Lancaster et al. (2007) icbm_spm transform. Grouping the peaks into the
deck's four regions (IFG, MFG, parietal, basal ganglia) is done by
anatomical label, see `PRADO_REGION_OF`.
"""
from __future__ import annotations

import numpy as np

NETWORK_PLAYERS = ("DR", "MD", "LANG")

# (x, y, z) Talairach, sphere radius mm -- from Xiao et al. run_get_dr_roi.m
PRADO_2011_TAL = {
    "general": [((-45, 35, 10), 5), ((-16, 1, 11), 5), ((-44, 26, 13), 5), ((-49, 22, 16), 5), ((-48, 10, 24), 5),
                ((-46, 15, 23), 5), ((-44, 5, 33), 5), ((-37, -59, 38), 10), ((-39, -57, 36), 10), ((-24, -71, 40), 6),
                ((-44, 12, 35), 5), ((-42, 10, 42), 5), ((39, 10, 40), 5), ((-42, -3, 39), 5), ((21, -69, 38), 5),
                ((-39, -6, 50), 7), ((-40, -6, 48), 6), ((-33, -8, 53), 5), ((-35, -5, 55), 5), ((-5, 3, 51), 8),
                ((-44, 4, 45), 5), ((20, -69, 41), 5), ((24, -64, 42), 5), ((-33, -14, 49), 5), ((-33, -14, 53), 5),
                ((24, -9, 58), 5)],
    "categorical": [((-47, 12, 23), 7), ((-39, -15, 47), 5), ((8, 6, 3), 7), ((-16, 7, 7), 5), ((-12, 0, 9), 5)],
    "relational": [((22, -7, 56), 7), ((-2, -1, 53), 5), ((-37, -56, 38), 5), ((-33, -61, 38), 5), ((24, -62, 40), 5),
                   ((-24, -69, 41), 5), ((20, -69, 41), 5), ((-9, -67, 41), 5)],
}

_ICBM_SPM = np.array([[0.9254, 0.0024, -0.0118, -1.0207],
                      [-0.0048, 0.9316, -0.0871, -1.7667],
                      [0.0152, 0.0883, 0.8924, 4.0926],
                      [0.0, 0.0, 0.0, 1.0]])


def tal2mni(xyz) -> np.ndarray:
    """Lancaster 2007 Talairach -> MNI (SPM), same as Xiao et al.'s tal2icbm_spm.m."""
    p = np.c_[np.atleast_2d(xyz), np.ones(len(np.atleast_2d(xyz)))]
    return (np.linalg.inv(_ICBM_SPM) @ p.T).T[:, :3]


def PRADO_REGION_OF(tal_xyz) -> str:
    """Coarse anatomical grouping of a Prado peak into the deck's 4 regions."""
    x, y, z = tal_xyz
    if abs(x) < 25 and z < 15 and y > -10:
        return "basal_ganglia"
    if y < -40:
        return "parietal"
    if z <= 30 and abs(x) >= 40:
        return "IFG"
    return "MFG"  # middle/medial frontal + precentral peaks


def prado_spheres_mni() -> dict[str, list[tuple[np.ndarray, float]]]:
    out: dict[str, list] = {}
    for peaks in PRADO_2011_TAL.values():
        for xyz, r in peaks:
            out.setdefault(PRADO_REGION_OF(xyz), []).append((tal2mni(xyz)[0], float(r)))
    return out


def build_region_masker(radius_scale: float = 1.0):
    """nilearn NiftiSpheresMasker over all Prado peaks (raw-BOLD path only;
    nilearn is an optional extra: pip install -e .[fmri])."""
    from nilearn.maskers import NiftiSpheresMasker

    spheres = [(c, r) for lst in prado_spheres_mni().values() for c, r in lst]
    return NiftiSpheresMasker(seeds=[tuple(c) for c, _ in spheres], radius=max(r for _, r in spheres) * radius_scale)
