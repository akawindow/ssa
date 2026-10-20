"""SRD-style Python prototype. NOT validated equivalent to the original Java SRD.

Follows the public implementation's moving median, constant boundary extension,
same-phase seasonal estimate, residual threshold and same-phase residual repair.
Uses NumPy median (averages the two central values for even counts); author helper
median semantics have NOT been independently checked. Paper's random Gaussian
repair differs from the currently inspected Java generate() implementation.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IMPLEMENTATION = 'srd_style_python_prototype_not_java_parity_verified'


def decompose_seasonal(x, period):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite 1-D series')
    if period < 2 or period % 2 or len(x) < 2 * period:
        raise ValueError('Prototype requires an even period and at least two cycles')
    half = period // 2
    trend = np.empty_like(x)
    medians = np.median(sliding_window_view(x, period), axis=1)
    # Positions follow the inspected Java Decomposition indexing.
    trend[half:half + len(medians)] = medians
    trend[:half] = trend[half]
    # Java constant_ext starts at N-half-1; preserve that boundary convention.
    trend[len(x)-half:] = trend[len(x)-half-1]
    detrended = x - trend
    seasonal = np.array([np.median(detrended[j::period]) for j in range(period)])
    seasonal -= np.median(seasonal)
    residual = detrended - seasonal[np.arange(len(x)) % period]
    return trend, seasonal, residual


def repair(x, period, k=3.0, max_iter=5):
    if k <= 0 or max_iter < 1:
        raise ValueError('Invalid repair configuration')
    out = np.array(x, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        trend, seasonal, residual = decompose_seasonal(out, period)
        sigma = residual.std()
        if sigma <= 1e-12:
            break
        bad = np.flatnonzero(np.abs(residual - residual.mean()) > k * sigma)
        if not len(bad):
            break
        for pos in bad:
            peers = np.arange(pos % period, len(out), period)
            peers = peers[peers != pos]
            out[pos] = trend[pos] + seasonal[pos % period] + np.median(residual[peers])
    return out
