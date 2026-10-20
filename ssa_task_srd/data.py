from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_series(config, root):
    if config['dataset'] == 'synthetic':
        rng = np.random.default_rng(config['seed'])
        t = np.arange(2400)
        p = config['period']
        values = 10 + .002*t + np.sin(2*np.pi*t/p) + .3*np.cos(4*np.pi*t/p) + rng.normal(0,.03,len(t))
        times = pd.date_range('2020-01-01', periods=len(t), freq='10min')
        return values, times, {'source':'explicit_synthetic_smoke_data','rows':len(values)}
    if config['dataset'] != 'tetouan':
        raise ValueError('Unsupported dataset; never silently substitute synthetic data')
    path = Path(root) / config['csv']
    frame = pd.read_csv(path)
    times = pd.DatetimeIndex(pd.to_datetime(frame['DateTime'], format='%m/%d/%Y %H:%M', errors='raise'))
    values = frame[config['column']].to_numpy(dtype=np.float64)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Timestamps must be ordered and unique; audit before repair')
    if not np.all(np.diff(times.asi8) == pd.Timedelta(minutes=10).value):
        raise ValueError('Irregular timestamps; do not silently fill gaps')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite observations')
    return values, times, {
        'source':'UCI Tetouan original CSV, no splicing', 'url':'https://doi.org/10.24432/C5B034',
        'license':'CC BY 4.0', 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows':len(frame), 'start':str(times[0]), 'end':str(times[-1]),
        'interval_minutes':10, 'duplicates':0, 'missing_values':int(frame.isna().sum().sum()),
        'reference':'Original observations, not guaranteed artifact-free ground truth',
    }


def fit_scale(values, train_end):
    train = np.asarray(values[:train_end], dtype=np.float64)
    std = float(train.std())
    if std <= 1e-12:
        raise ValueError('Training series has zero variance')
    return float(train.mean()), std


def inject(reference_z, config):
    """One reproducible timeline realization shared by all overlapping windows."""
    rng = np.random.default_rng(config.get('corruption_seed', config['seed']))
    out = np.array(reference_z, copy=True)
    out += rng.normal(0, config['noise_std'], len(out))
    mask = np.zeros(len(out), dtype=bool)
    starts = np.flatnonzero(rng.random(len(out)) < config['outlier_rate'])
    for start in starts:
        stop = min(len(out), start + config['outlier_length'])
        out[start:stop] += rng.choice([-1.0,1.0]) * config['outlier_scale']
        mask[start:stop] = True
    return out, mask


def observed_input(reference_z, config):
    """Original-observation experiments bypass error injection entirely."""
    if config.get('input_condition') == 'reference':
        if config.get('noise_std',0)!=0 or config.get('outlier_rate',0)!=0:
            raise ValueError('Reference condition requires zero added noise and error rates')
        return np.array(reference_z,copy=True),np.zeros(len(reference_z),dtype=bool)
    return inject(reference_z,config)


def make_windows(reference_z, observed_z, mask, config):
    n = len(reference_z)
    train_end, val_end = int(.7*n), int(.85*n)
    history, horizon, stride = config['history'], config['horizon'], config['stride']
    if min(history,horizon,stride) < 1:
        raise ValueError('Window dimensions must be positive')
    all_origins = np.arange(history, n-horizon+1, stride, dtype=int)
    conditions = {
        'train': all_origins[all_origins + horizon <= train_end],
        'val': all_origins[(all_origins >= train_end) & (all_origins+horizon <= val_end)],
        'test': all_origins[all_origins >= val_end],
    }
    if 'test' not in config.get('evaluation_splits', ['val', 'test']):
        conditions.pop('test')
    result = {}
    for split, origins in conditions.items():
        limit = config.get('max_'+split)
        if limit and len(origins)>limit:
            origins = origins[np.unique(np.linspace(0,len(origins)-1,limit).astype(int))]
        if not len(origins):
            raise ValueError(f'No windows in {split}')
        result[split] = {
            'origins':origins,
            'x':np.array([observed_z[o-history:o] for o in origins]),
            'reference':np.array([reference_z[o-history:o] for o in origins]),
            'y':np.array([reference_z[o:o+horizon] for o in origins]),
            'outlier_mask':np.array([mask[o-history:o] for o in origins]),
        }
    return result, {'train_end':train_end,'val_end':val_end,'test_end':n}
