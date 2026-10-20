"""Training-only SSA waveforms, spectra, weighted correlations and cutoff checks."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.data import load_series, fit_scale
from ssa_task_srd.ssa_structure import decomposition_details, weighted_correlation, spectral_features, spectrum, candidate_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root/args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = json.loads((root/'configs/ssa_structure_protocol.json').read_text(encoding='utf-8'))
    config = json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    assert config['input_condition'] == 'reference' and config['repair_backend'] == 'none'
    assert config['noise_std'] == config['outlier_rate'] == 0
    values, times, audit = load_series(config, root)
    train_end = int(.7*len(values))
    mean, std = fit_scale(values, train_end)
    train = (values[:train_end]-mean)/std
    origins = np.arange(config['history'], train_end-config['horizon']+1, config['stride'], dtype=int)
    if config.get('max_train') and len(origins) > config['max_train']:
        origins = origins[np.unique(np.linspace(0, len(origins)-1, config['max_train']).astype(int))]
    histories = np.array([train[o-config['history']:o] for o in origins])
    nshow = protocol['display_components']
    nsearch = protocol['pair_search_components']
    cutoff = protocol['existing_cutoff']
    thresholds = protocol['candidate_pair_thresholds']
    metric_rows, pair_rows, cutoff_rows, singular_rows, correlations = [], [], [], [], []
    all_components, all_singular, example_series = [], [], {}
    example_indices = [0, (len(origins)-1)//2, len(origins)-1]
    max_identity_error = 0.
    for index, (origin, x) in enumerate(zip(origins, histories)):
        c, singular, weights = decomposition_details(x, protocol['embedding'])
        identity_error = float(np.max(np.abs(c.sum(axis=0)-x)))
        assert identity_error < 1e-10
        max_identity_error = max(max_identity_error, identity_error)
        all_components.append(c)
        all_singular.append(singular)
        corr = weighted_correlation(c[:nsearch], weights)
        correlations.append(corr)
        stats = []
        for i in range(nsearch):
            f = spectral_features(c[i], 144, protocol['daily_harmonics_cycles_per_day'])
            stats.append(f)
            metric_rows.append(dict(window=index, origin=int(origin), component=f'C{i+1}', **f))
        raw = spectral_features(x)
        residual = x-c[:cutoff].sum(axis=0)
        res = spectral_features(residual)
        for name, series in [('raw', x), ('prefix_6', x-residual), ('residual_after_6', residual)]:
            metric_rows.append(dict(window=index, origin=int(origin), component=name, **spectral_features(series)))
        candidates_crossing = []
        for i in range(nsearch):
            for j in range(i+1, nsearch):
                is_candidate = candidate_pair(singular[i], singular[j], corr[i, j],
                                               stats[i]['peak_bin'], stats[j]['peak_bin'], thresholds)
                cross = i < cutoff <= j
                pair_rows.append({'window': index, 'origin': int(origin), 'left': f'C{i+1}',
                    'right': f'C{j+1}', 'signed_wcor': float(corr[i, j]), 'abs_wcor': abs(float(corr[i, j])),
                    'relative_singular_value_gap': float(abs(singular[i]-singular[j])/singular[i]),
                    'same_positive_peak_bin': bool(stats[i]['peak_bin'] > 0 and stats[i]['peak_bin'] == stats[j]['peak_bin']),
                    'candidate_pair': is_candidate, 'crosses_cutoff_6': cross})
                if is_candidate and cross:
                    candidates_crossing.append(f'C{i+1}+C{j+1}')
        cross_matrix = np.abs(corr[:cutoff, cutoff:])
        cross_i, cross_j = np.unravel_index(np.argmax(cross_matrix), cross_matrix.shape)
        cutoff_rows.append({'window': index, 'origin': int(origin),
            'history_start': str(times[origin-config['history']]), 'history_end': str(times[origin-1]),
            'first_6_trajectory_energy_fraction': float(np.sum(singular[:cutoff]**2)/np.sum(singular**2)),
            'residual_centered_rms_z': res['centered_rms_z'], 'raw_centered_rms_z': raw['centered_rms_z'],
            'residual_to_raw_centered_rms': res['centered_rms_z']/raw['centered_rms_z'],
            'residual_daily_amplitude_z': res['daily_amplitude_z'], 'raw_daily_amplitude_z': raw['daily_amplitude_z'],
            'residual_to_raw_daily_amplitude': res['daily_amplitude_z']/raw['daily_amplitude_z'] if raw['daily_amplitude_z'] > 1e-12 else np.nan,
            'residual_daily_harmonic_power_fraction': res['daily_harmonic_power_fraction'],
            'residual_peak_period_hours': res['peak_period_hours'],
            'c6_c7_abs_wcor': abs(float(corr[cutoff-1, cutoff])),
            'c6_c7_candidate': candidate_pair(singular[cutoff-1], singular[cutoff], corr[cutoff-1, cutoff],
                                             stats[cutoff-1]['peak_bin'], stats[cutoff]['peak_bin'], thresholds),
            'max_cross_cutoff_abs_wcor': float(cross_matrix[cross_i, cross_j]),
            'max_cross_cutoff_pair': f'C{cross_i+1}+C{cross_j+cutoff+1}',
            'candidate_cross_cutoff_count': len(candidates_crossing),
            'candidate_cross_cutoff_pairs': ';'.join(candidates_crossing)})
        for i, value in enumerate(singular):
            singular_rows.append({'window': index, 'origin': int(origin), 'component': i+1,
                'singular_value': value, 'trajectory_energy_fraction': value**2/np.sum(singular**2)})
        if index in example_indices:
            example_series[index] = dict(raw=x, residual=residual, components=c,
                origin=int(origin), start=str(times[origin-config['history']]), end=str(times[origin-1]))
    frame = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    cuts = pd.DataFrame(cutoff_rows)
    frame.to_csv(out/'component_metrics.csv', index=False)
    pairs.to_csv(out/'pair_metrics.csv', index=False)
    cuts.to_csv(out/'cutoff_diagnostics.csv', index=False)
    pd.DataFrame(singular_rows).to_csv(out/'singular_values.csv', index=False)
    summary = frame.groupby('component', sort=False).agg(
        median_peak_period_hours=('peak_period_hours', 'median'),
        median_centered_rms_z=('centered_rms_z', 'median'),
        median_daily_harmonic_power_fraction=('daily_harmonic_power_fraction', 'median'),
        median_daily_amplitude_z=('daily_amplitude_z', 'median'))
    # Rank labels may change meaning between histories. A median period can lie
    # between observed peaks, so also retain the actual peak distribution/mode.
    peak_distribution = frame.groupby(['component', 'peak_period_hours'], sort=False).size().rename('window_count').reset_index()
    peak_distribution.to_csv(out/'peak_period_distribution.csv', index=False)
    modes = peak_distribution.sort_values('window_count', ascending=False).drop_duplicates('component').set_index('component')
    summary['modal_peak_period_hours'] = modes['peak_period_hours']
    summary['modal_peak_window_count'] = modes['window_count']
    summary.to_csv(out/'component_summary.csv')
    pair_summary = pairs.groupby(['left', 'right'], sort=False).agg(
        candidate_windows=('candidate_pair', 'sum'), median_abs_wcor=('abs_wcor', 'median'),
        median_relative_singular_gap=('relative_singular_value_gap', 'median'),
        crosses_cutoff_6=('crosses_cutoff_6', 'first')).reset_index()
    pair_summary.sort_values(['candidate_windows', 'median_abs_wcor'], ascending=False).to_csv(out/'pair_summary.csv', index=False)
    median_corr = np.median(np.abs(correlations), axis=0)
    pd.DataFrame(median_corr, index=[f'C{i+1}' for i in range(nsearch)],
                 columns=[f'C{i+1}' for i in range(nsearch)]).to_csv(out/'median_absolute_wcor.csv')
    np.savez_compressed(out/'training_decompositions.npz', histories=histories, origins=origins,
                         components=np.array(all_components), singular_values=np.array(all_singular),
                         signed_wcor=np.array(correlations))
    plot(out, example_series, np.array(all_singular), median_corr, cuts, nshow, cutoff)
    audit.update({'scope': protocol['scope'], 'training_boundary': train_end,
        'standardization_fit_range': [0, train_end], 'scale_mean': mean, 'scale_std': std,
        'window_count': len(origins), 'history_length': config['history'], 'embedding': protocol['embedding'],
        'origins': origins.tolist(), 'example_window_indices': example_indices,
        'max_full_reconstruction_difference_z': max_identity_error,
        'no_injection_function_called': True, 'repair_executed': False, 'models_trained': 0,
        'validation_histories_constructed': False, 'test_histories_constructed': False, 'targets_constructed': False,
        'frequency_resolution_cycles_per_day': 144/config['history'],
        'meaning': 'Descriptive statistics of original training histories; not denoising accuracy or evidence of forecast improvement'})
    result = {'train_windows': len(origins),
        'median_first_6_trajectory_energy_fraction': float(cuts.first_6_trajectory_energy_fraction.median()),
        'median_residual_to_raw_centered_rms': float(cuts.residual_to_raw_centered_rms.median()),
        'median_residual_to_raw_daily_amplitude': float(cuts.residual_to_raw_daily_amplitude.median()),
        'median_residual_daily_harmonic_power_fraction': float(cuts.residual_daily_harmonic_power_fraction.median()),
        'windows_with_any_candidate_crossing_cutoff': int((cuts.candidate_cross_cutoff_count > 0).sum()),
        'windows_with_c6_c7_candidate': int(cuts.c6_c7_candidate.sum()),
        'median_c6_c7_abs_wcor': float(cuts.c6_c7_abs_wcor.median()),
        'c6_c7_candidate_peak_period_counts': frame[(frame.component == 'C6') & frame.window.isin(
            pairs[(pairs.left == 'C6') & (pairs.right == 'C7') & pairs.candidate_pair].window)
            ].peak_period_hours.value_counts().to_dict(),
        'leading_candidate_pairs': pair_summary.sort_values(['candidate_windows', 'median_abs_wcor'], ascending=False).head(12).to_dict('records')}
    for name, value in [('protocol', protocol), ('config', config), ('data_audit', audit), ('summary', result)]:
        (out/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    (out/'metadata.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': commit, 'elapsed_seconds': time.perf_counter()-started,
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('Training-only structure diagnostics completed; no targets, corruption, repair or forecasting.')


def plot(out, examples, singular, median_corr, cuts, nshow, cutoff):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f'C{i+1}' for i in range(nshow)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    relative = singular**2/np.sum(singular**2, axis=1, keepdims=True)
    m = np.median(relative[:, :24], axis=0)
    lo, hi = np.quantile(relative[:, :24], [.25, .75], axis=0)
    axes[0].semilogy(np.arange(1, 25), m, marker='.', label='Median of 36 histories')
    axes[0].fill_between(np.arange(1, 25), lo, hi, alpha=.2, label='Interquartile range')
    axes[0].axvline(cutoff+.5, color='red', ls='--')
    axes[0].set(xlabel='Component rank', ylabel='Trajectory energy fraction', title='Singular spectrum (not noise probability)')
    axes[0].legend(fontsize=8)
    im = axes[1].imshow(median_corr[:nshow, :nshow], vmin=0, vmax=1, cmap='viridis')
    axes[1].set_xticks(range(nshow), labels, rotation=90)
    axes[1].set_yticks(range(nshow), labels)
    axes[1].axvline(cutoff-.5, color='red', ls='--'); axes[1].axhline(cutoff-.5, color='red', ls='--')
    axes[1].set_title('Median absolute w-correlation')
    fig.colorbar(im, ax=axes[1], shrink=.7)
    axes[2].plot(cuts.window, cuts.residual_to_raw_centered_rms, marker='.', label='Centered RMS: residual / raw')
    axes[2].plot(cuts.window, cuts.residual_to_raw_daily_amplitude, marker='.', label='24 h amplitude: residual / raw')
    axes[2].set(xlabel='Training history index', ylabel='Ratio', title='What remains after removing C1-C6?')
    axes[2].legend(fontsize=8)
    fig.savefig(out/'structure_overview.png', dpi=150); plt.close(fig)

    for spectra in [False, True]:
        fig, axes = plt.subplots(nshow+2, len(examples), figsize=(15, 21), layout='constrained')
        for col, (index, example) in enumerate(examples.items()):
            rows = [('Raw', example['raw'])]+[(f'C{i+1}', example['components'][i]) for i in range(nshow)]+[('Residual after C6', example['residual'])]
            for row, (name, series) in enumerate(rows):
                ax = axes[row, col]
                if spectra:
                    f, p = spectrum(series)
                    power_total = p[1:].sum()
                    ax.plot(f, p/power_total if power_total > 0 else p, lw=.9)
                    ax.set_xlim(0, 8)
                    for h in [1, 2, 3]: ax.axvline(h, lw=.6, color='grey', ls=':')
                    stats = spectral_features(series)
                    ax.text(.98, .8, f"Peak: {stats['peak_period_hours']:.1f} h", ha='right', fontsize=8, transform=ax.transAxes)
                else:
                    ax.plot(np.arange(len(series))/144, series, lw=.9)
                    ax.axhline(0, lw=.4, color='grey')
                if col == 0: ax.set_ylabel(name, fontsize=9)
                if row == 0: ax.set_title(f"Train index {index}; {example['start'][:10]} to {example['end'][:10]}", fontsize=10)
                if row == nshow+1: ax.set_xlabel('Cycles per day' if spectra else 'Days within observed history')
                ax.tick_params(labelsize=8)
        fig.suptitle('Mean-subtracted spectra; power normalized within each series; view limited to 0-8 cycles/day'
                     if spectra else 'Original training histories and elementary SSA reconstructions; standardized units', fontsize=12)
        fig.savefig(out/('component_spectra.png' if spectra else 'component_waveforms.png'), dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 9), layout='constrained')
    for row, (index, example) in enumerate(examples.items()):
        x, residual = example['raw'], example['residual']
        t = np.arange(len(x))/144
        axes[row, 0].plot(t, x, color='black', lw=1, label='Raw')
        axes[row, 0].plot(t, x-residual, lw=1, label='C1-C6')
        axes[row, 0].plot(t, residual, lw=.9, label='Residual after C6')
        axes[row, 0].set(title=f'Training history {index}', xlabel='Days', ylabel='Standardized value')
        for name, series in [('Raw', x), ('Residual after C6', residual)]:
            f, p = spectrum(series)
            axes[row, 1].plot(f, p, label=name)
        axes[row, 1].set(xlim=(0, 8), xlabel='Cycles per day', ylabel='Mean-subtracted spectral power')
        for h in [1, 2, 3]: axes[row, 1].axvline(h, lw=.6, color='grey', ls=':')
        for ax in axes[row]: ax.legend(fontsize=8)
    fig.suptitle('Residual structure; regular variations are not automatically measurement noise')
    fig.savefig(out/'residual_examples.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
