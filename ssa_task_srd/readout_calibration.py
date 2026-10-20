"""Chronological calibration of Ridge on frozen input representations."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def expanding_folds(n, first_train_fraction=.4, n_splits=3):
    if n_splits < 1 or not 0 < first_train_fraction < 1:
        raise ValueError('Invalid chronological fold settings')
    first = int(n*first_train_fraction)
    edges = np.linspace(first, n, n_splits+1).astype(int)
    if first < 2 or np.any(np.diff(edges) < 1):
        raise ValueError('Too few histories for chronological folds')
    return [(np.arange(left), np.arange(left, right)) for left, right in zip(edges[:-1], edges[1:])]


def pipeline(alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Ridge alpha must be positive')
    return Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=float(alpha), solver='svd'))])


def select_alpha(x, y, folds, alphas):
    """Only outer-training arrays are accepted. Scalers fit independently per fold."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    alphas = sorted(set(float(a) for a in alphas))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not alphas:
        raise ValueError('Expected corresponding input/target matrices and alpha candidates')
    rows = []
    for alpha in alphas:
        for fold, (train, val) in enumerate(folds):
            if max(train) >= min(val) or len(set(train) & set(val)):
                raise ValueError('Inner validation must follow its training prefix')
            model = pipeline(alpha).fit(x[train], y[train])
            prediction = model.predict(x[val])
            rows.append({'fold': fold, 'alpha': alpha, 'sse_z': float(np.square(prediction-y[val]).sum()),
                         'point_count': int(y[val].size), 'mse_z': float(np.square(prediction-y[val]).mean())})
    scores = {alpha: sum(r['sse_z'] for r in rows if r['alpha'] == alpha)/
              sum(r['point_count'] for r in rows if r['alpha'] == alpha) for alpha in alphas}
    best = min(scores.values())
    tolerance = max(1e-12, abs(best)*1e-10)
    chosen = max(alpha for alpha, score in scores.items() if score <= best+tolerance)
    return chosen, rows, scores
