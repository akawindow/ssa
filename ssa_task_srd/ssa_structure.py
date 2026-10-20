"""Descriptive SSA diagnostics. These statistics do not identify noise."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import svd
from scipy.signal import periodogram


def decomposition_details(x, embedding):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or not 2 <= embedding < len(x):
        raise ValueError('Expected finite history and valid embedding')
    trajectory = sliding_window_view(x, embedding).T.copy()
    u, singular, vh = svd(trajectory, full_matrices=False, check_finite=False,
                           lapack_driver='gesdd')
    counts = np.convolve(np.ones(embedding), np.ones(trajectory.shape[1]))
    components = np.array([s*np.convolve(u[:, i], vh[i])/counts
                           for i, s in enumerate(singular)])
    return components, singular, counts


def weighted_correlation(components, weights):
    """SSA weighted cosine similarities, with no extra mean subtraction."""
    c = np.asarray(components, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if c.ndim != 2 or w.shape != (c.shape[1],) or np.any(w <= 0):
        raise ValueError('Expected component rows and positive anti-diagonal counts')
    gram = (c*w) @ c.T
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    denominator = norms[:, None]*norms[None, :]
    return np.clip(np.divide(gram, denominator, out=np.zeros_like(gram),
                             where=denominator > 0), -1, 1)


def spectrum(x, samples_per_day=144):
    frequency, power = periodogram(np.asarray(x, dtype=np.float64),
                                    fs=samples_per_day, window='boxcar',
                                    detrend='constant', scaling='spectrum')
    return frequency, power


def spectral_features(x, samples_per_day=144, harmonics=(1, 2, 3)):
    x = np.asarray(x, dtype=np.float64)
    frequency, power = spectrum(x, samples_per_day)
    total = float(power[1:].sum())
    peak = int(1+np.argmax(power[1:])) if total > 1e-28 else 0
    step = samples_per_day/len(x)
    daily_mask = np.zeros(len(power), dtype=bool)
    for h in harmonics:
        daily_mask |= np.abs(frequency-h) < step/2 + 1e-12
    t = np.arange(len(x))/samples_per_day
    basis = np.column_stack([np.ones(len(x)), np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
    coef = np.linalg.lstsq(basis, x, rcond=None)[0]
    return {
        'rms_z': float(np.sqrt(np.mean(x*x))),
        'centered_rms_z': float(np.std(x)),
        'peak_bin': peak,
        'peak_cycles_per_day': float(frequency[peak]) if peak else np.nan,
        'peak_period_hours': float(24/frequency[peak]) if peak else np.nan,
        'daily_harmonic_power_fraction': float(power[daily_mask].sum()/total) if total > 1e-28 else np.nan,
        'daily_amplitude_z': float(np.hypot(coef[1], coef[2])),
    }


def candidate_pair(singular_i, singular_j, wcor, peak_i, peak_j, thresholds):
    gap = float(abs(singular_i-singular_j)/max(singular_i, singular_j)) if max(singular_i, singular_j) > 0 else np.inf
    same_peak = peak_i > 0 and peak_i == peak_j
    return bool(abs(wcor) >= thresholds['min_absolute_wcor']
                and gap <= thresholds['max_relative_singular_value_gap']
                and (same_peak or not thresholds['require_same_positive_peak_bin']))
