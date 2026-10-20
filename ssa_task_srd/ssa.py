"""Basic trajectory-matrix SSA with diagonal averaging (not a learned SVD)."""
import numpy as np
from scipy.linalg import svd
from numpy.lib.stride_tricks import sliding_window_view


def decompose(x, embedding, rank):
    """Return rank individual RCs plus one exact residual group, shape (rank+1, N)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('SSA needs one finite one-dimensional history window')
    if not 2 <= embedding <= len(x) - 1:
        raise ValueError('Invalid SSA embedding length')
    trajectory = sliding_window_view(x, embedding).T.copy()
    if not 1 <= rank <= min(trajectory.shape):
        raise ValueError('Invalid SSA retained rank')
    u, s, vh = svd(trajectory, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    counts = np.convolve(np.ones(embedding), np.ones(trajectory.shape[1]))
    components = np.array([s[i] * np.convolve(u[:, i], vh[i]) / counts for i in range(rank)])
    residual = x - components.sum(axis=0)
    return np.vstack([components, residual])


def ordinary(components):
    """Fixed-rank SSA baseline: omit only the residual group."""
    return np.asarray(components)[:-1].sum(axis=0)
