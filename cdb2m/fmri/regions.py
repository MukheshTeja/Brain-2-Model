"""Phase B scaffold — NOT implemented yet.

Region masks: spheres around Prado et al. 2011 meta-analysis coordinates
(IFG, MFG, parietal, basal ganglia), in MNI space, via nilearn. These 4
regions are RQ2's Shapley players (nullladder.shapley.exact_shapley over
all 16 coalitions).
"""
from __future__ import annotations

# MNI coordinates (x, y, z) from Prado, Chadha & Booth (2011) deductive
# reasoning meta-analysis. PLACEHOLDER — replace with the paper's actual
# peak coordinates before running Phase B.
PRADO_2011_COORDINATES = {
    "IFG": (-45, 20, 10),
    "MFG": (-40, 30, 30),
    "parietal": (-30, -60, 45),
    "basal_ganglia": (-15, 10, 5),
}

SPHERE_RADIUS_MM = 8


def build_region_masker(coordinates: dict[str, tuple[float, float, float]] = PRADO_2011_COORDINATES, radius_mm: float = SPHERE_RADIUS_MM):
    """Returns a nilearn NiftiSpheresMasker built from `coordinates`. Deferred
    import so nilearn (fmri extra) isn't required for Phase A."""
    from nilearn.maskers import NiftiSpheresMasker

    seeds = list(coordinates.values())
    return NiftiSpheresMasker(seeds=seeds, radius=radius_mm, standardize=True)
