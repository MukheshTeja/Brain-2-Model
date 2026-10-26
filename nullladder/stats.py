"""Spectral and covariance statistics shared by every surrogate generator and its tests.

Every function here operates on unconditional (time-averaged) statistics of a
[T, C] signal — never on trial-locked / condition-specific statistics. That
constraint is load-bearing for the Null Ladder: a surrogate that leaked a
trial-locked mean would trivially carry task structure no matter which rung
produced it.
"""
from __future__ import annotations

import numpy as np


def per_channel_power_spectrum(x: np.ndarray) -> np.ndarray:
    """Periodogram power spectrum per channel, on DEMEANED data. x: [T, C] -> [F, C].

    Demeaning first matters whenever two DIFFERENT finite realizations are
    being compared (e.g. structured vs. hollow brain): each realization's
    raw sample mean is itself a noisy, realization-specific quantity, and a
    DC bin computed without demeaning is dominated by that noise rather
    than by the shared dynamics the comparison actually cares about. For a
    surrogate compared against the SAME real signal's own DC term (R1-R3,
    which preserve DC exactly), demeaning changes nothing.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    xf = np.fft.rfft(xc, axis=0)
    return (xf.conj() * xf).real / x.shape[0]


def zero_lag_covariance(x: np.ndarray) -> np.ndarray:
    """Cross-channel covariance at lag 0. x: [T, C] -> [C, C]."""
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / x.shape[0]


def lagged_cross_covariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Cross-channel covariance at lags 0..max_lag. x: [T, C] -> [max_lag+1, C, C].

    out[k, i, j] = Cov(x_i[t], x_j[t+k]) averaged over t.
    """
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    out = np.zeros((max_lag + 1, x.shape[1], x.shape[1]))
    for k in range(max_lag + 1):
        out[k] = (xc[: T - k].T @ xc[k:]) / (T - k)
    return out


def cross_spectral_density(x: np.ndarray) -> np.ndarray:
    """Full cross-spectral density matrix per frequency. x: [T, C] -> [F, C, C] complex."""
    xf = np.fft.rfft(x, axis=0)  # [F, C]
    F, C = xf.shape
    return np.einsum("fi,fj->fij", xf, xf.conj()) / x.shape[0]


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Per-channel autocorrelation (normalized), lags 0..max_lag. x: [T, C] -> [max_lag+1, C]."""
    cc = lagged_cross_covariance(x, max_lag)
    var = np.diagonal(cc[0])
    diag = np.stack([np.diagonal(cc[k]) for k in range(max_lag + 1)], axis=0)
    return diag / var[None, :]


def total_variance(x: np.ndarray) -> np.ndarray:
    """Per-channel variance. x: [T, C] -> [C]."""
    return x.var(axis=0)


def effective_dimensionality(x: np.ndarray) -> float:
    """Participation-ratio dimensionality of the zero-lag covariance spectrum."""
    cov = zero_lag_covariance(x)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    return float((eigvals.sum() ** 2) / (eigvals**2).sum())


def spectrum_match_error(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized L2 error between per-channel power spectra of x and y.

    Uses the RAW periodogram, which is appropriate when y is a surrogate
    DERIVED from x (R0-R3): phase randomization copies x's exact per-bin
    magnitude into y, so matching bins are compared against matching bins.
    For comparing two INDEPENDENTLY drawn realizations (e.g. structured vs.
    hollow brain) use smoothed_spectrum_match_error instead — raw
    periodogram bins have high variance (each bin is roughly
    chi-squared(2)) even when both realizations share the same underlying
    spectral density, so this function would overstate the mismatch.
    """
    sx, sy = per_channel_power_spectrum(x), per_channel_power_spectrum(y)
    return float(np.linalg.norm(sx - sy) / (np.linalg.norm(sx) + 1e-12))


def smoothed_spectrum_match_error(x: np.ndarray, y: np.ndarray, bin_width: int = 10) -> float:
    """Like spectrum_match_error, but averages the periodogram over groups
    of `bin_width` adjacent frequency bins first. The raw periodogram is
    not a consistent estimator of the power spectral density (its
    per-bin variance doesn't shrink as T grows); this boxcar-smoothed
    version is the right comparison for two INDEPENDENTLY drawn
    realizations that are only expected to share the same underlying
    spectral shape, not matching bins."""
    sx, sy = per_channel_power_spectrum(x), per_channel_power_spectrum(y)
    n_bins = (sx.shape[0] // bin_width) * bin_width
    sx_smooth = sx[:n_bins].reshape(-1, bin_width, sx.shape[1]).mean(axis=1)
    sy_smooth = sy[:n_bins].reshape(-1, bin_width, sy.shape[1]).mean(axis=1)
    return float(np.linalg.norm(sx_smooth - sy_smooth) / (np.linalg.norm(sx_smooth) + 1e-12))


def covariance_match_error(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized Frobenius error between zero-lag covariance matrices."""
    cx, cy = zero_lag_covariance(x), zero_lag_covariance(y)
    return float(np.linalg.norm(cx - cy) / (np.linalg.norm(cx) + 1e-12))


def lagged_cross_covariance_norm(x: np.ndarray, max_lag: int, skip_zero_lag: bool = True) -> float:
    """Average Frobenius norm of lagged (nonzero-lag) cross-covariance matrices.

    Used to confirm R1/R2 destroy lagged cross-structure while R3 preserves it.
    """
    cc = lagged_cross_covariance(x, max_lag)
    start = 1 if skip_zero_lag else 0
    return float(np.mean([np.linalg.norm(cc[k]) for k in range(start, max_lag + 1)]))
