"""Posthoc numerical-tie and zero-component checks; never changes source scores."""
import os
for k in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(k, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ssa_task_srd.classification import build_classifier, build_gate
from ssa_task_srd.learning import tensorflow, weight_digest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(__file__).resolve().parents[1]; source = root/a.run
    out = root/a.out; out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source/'protocol.json').read_text()); run = root/protocol['source_run']
    cfg = json.loads((run/'protocol.json').read_text()); audit = json.loads((run/'data_audit.json').read_text())
    probes = pd.read_csv(source/'component_probes.csv'); fit = np.array(audit['indices']['fit'])
    tf = tensorflow(101); rows = []
    for seed in protocol['seeds']:
        with np.load(source/f'features_weights_seed{seed}.npz') as z:
            weights = z['weights'].astype(np.float64)
        means = weights[fit, :32].mean(axis=0)
        q = probes[(probes.seed == seed) & (probes.role == 'assessment')].set_index('component')
        accuracy = [q.loc[f'C{i+1}', 'accuracy'] for i in range(32)]
        tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
        folder = run/f'task_full_seed{seed}'; proxy.load_weights(str(folder/'proxy.weights.h5'))
        gate = build_gate(tf, proxy, 'task_full', cfg, len(fit))
        with np.load(folder/'gate.npz') as z:
            gate.restore([z[f'w{i}'] for i in range(len(z.files))])
        before = (weight_digest(proxy.get_weights()), weight_digest(gate.weights()))
        zero_features = proxy.layers[0](np.zeros((1, cfg['length']), np.float32), training=False)
        zero_retention = float(gate.head(zero_features, training=False).numpy()[0, 0])
        assert before == (weight_digest(proxy.get_weights()), weight_digest(gate.weights()))
        rows.append(dict(seed=seed, zero_input_retention=zero_retention,
            C10_to_C32_max_mean_difference_from_zero=float(np.max(np.abs(means[9:]-zero_retention))),
            full_precision_spearman=float(spearmanr(means, accuracy).statistic),
            rounded_1e4_spearman=float(spearmanr(np.round(means, 4), accuracy).statistic),
            rounded_1e4_unique_weight_levels=int(len(np.unique(np.round(means, 4))))))
    pd.DataFrame(rows).to_csv(out/'zero_and_ties.csv', index=False)
    (out/'scope.json').write_text(json.dumps(dict(posthoc=True,
        reason='Near-tied weak-component weights make rank correlations sensitive to float32 reduction order. Recompute float64 means and round to 1e-4; preserve original raw correlations.',
        zero_input='Frozen encoder+head on zero component; no labels or training.',
        primary_scores_unchanged=True), indent=2), encoding='utf-8')
    w = pd.read_csv(source/'component_weights.csv')
    fitw = w[w.role == 'fit'].groupby('component')[['retention_mean', 'component_rms']].mean()
    p = probes[probes.role == 'assessment'].copy()
    p = p[p.component != 'raw']; p['component'] = p.component.str[1:].astype(int)
    summary = p.groupby('component').accuracy.agg(['mean', 'std']).join(fitw)
    summary.to_csv(out/'component_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, layout='constrained')
    axes[0].plot(summary.index, summary.retention_mean, 'o-')
    axes[0].set_ylim(0, 1); axes[0].set_ylabel('Fit mean retention (3 seeds)')
    axes[1].errorbar(summary.index, 100*summary['mean'], yerr=100*summary['std'], fmt='o-', capsize=2)
    axes[1].axhline(100*max(audit['class_counts']['assessment'])/audit['counts']['assessment'], linestyle='--', color='grey', label='Majority reference')
    axes[1].set_ylim(0, 100); axes[1].set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    axes[1].set_ylabel('Explored assessment accuracy (%)'); axes[1].set_xlabel('SSA component rank (not a physical source ID)')
    axes[1].legend()
    fig.suptitle('Frozen task encoder; separate fit-only linear probes; no CNN retraining')
    fig.savefig(out/'component_diagnostic.png', dpi=160); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
