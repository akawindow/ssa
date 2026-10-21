"""Inference/record audit and plots for a completed immutable classification run."""
import os
for key in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.classification import build_classifier, build_gate, predict, scores
from ssa_task_srd.learning import tensorflow, weight_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference-run', default=None)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    run = root/args.run; out = root/args.out; out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((run/'protocol.json').read_text()); audit = json.loads((run/'data_audit.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text()); frame = pd.read_csv(run/'metrics.csv')
    dataset_path = root/'data/forda/FordA_TRAIN.tsv'
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == audit['sha256']
    data = np.loadtxt(dataset_path, delimiter='\t'); labels = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    cache = np.load(root/audit['cache']); c = cache['components']; fixed = cache['fixed']
    inputs = {'raw_pretrained': x, 'raw_continue': x, 'half_continue': x*.5,
              'ssa_energy_continue': fixed, 'global_gate': x, 'task_full': c}
    fingerprints = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in run.rglob('*') if p.is_file()}
    rows = []; paired = []; shape_rows = []
    tf = tensorflow(101)
    for item in manifest:
        folder = run/item['folder']; branch = item['branch']; seed = item['seed']
        tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
        proxy.load_weights(str(folder/'proxy.weights.h5'))
        training = json.loads((folder/'training.json').read_text())
        expected = training['initial_for_branches_sha256'] if branch == 'raw_pretrained' else training['final_proxy_sha256']
        assert weight_digest(proxy.get_weights()) == expected
        gate = build_gate(tf, proxy, branch, cfg, audit['counts']['fit']) if branch in ['global_gate', 'task_full'] else None
        if gate:
            with np.load(folder/'gate.npz') as z:
                gate.restore([z[f'w{i}'] for i in range(len(z.files))])
            assert weight_digest(gate.weights()) == training['final_head_sha256']
        saved = np.load(folder/'logits.npz')
        for role, indices in audit['indices'].items():
            ix = np.array(indices)
            logits = predict(proxy, gate, inputs[branch][ix], cfg['batch_size'])
            np.testing.assert_allclose(logits, saved[role], atol=2e-6, rtol=2e-6)
            reported = frame[(frame.branch == branch) & (frame.seed == seed) & (frame.role == role)].iloc[0]
            sc = scores(logits, labels[ix])
            for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'cross_entropy']:
                np.testing.assert_allclose(sc[metric], reported[metric], atol=1e-12)
            if role == 'selection':
                np.testing.assert_allclose(sc['cross_entropy'], training['best_selection_cross_entropy'], atol=1e-7)
            rows.append(dict(branch=branch, seed=seed, role=role, max_abs_logit_difference=float(np.max(np.abs(logits-saved[role]))), metrics_match=True))
        if branch == 'task_full':
            # Geometry only: this does not score a new scalar-substitution predictor.
            for role, indices in audit['indices'].items():
                ix = np.array(indices)
                rebuilt = np.concatenate([gate.reconstruct(c[ix[s:s+64]])[0].numpy() for s in range(0, len(ix), 64)])
                raw = x[ix].astype(np.float64); rebuilt = rebuilt.astype(np.float64)
                scale = (raw*rebuilt).sum(axis=1)/(raw*raw).sum(axis=1)
                difference = rebuilt-scale[:, None]*raw
                shape_rows.append(dict(seed=seed, role=role, closest_per_sample_scale_mean=float(scale.mean()),
                    closest_scale_min=float(scale.min()), closest_scale_max=float(scale.max()),
                    shape_residual_rms_over_raw_rms=float(np.sqrt(np.square(difference).mean()/np.square(raw).mean())),
                    scope='Posthoc geometry only; no future/class labels used; not a denoising error or new predictor score'))
        print('Verified', branch, seed, flush=True)
    ix = np.array(audit['indices']['assessment'])
    for seed in cfg['seeds']:
        reference = root/args.reference_run if args.reference_run else run
        raw = np.load(reference/f'raw_continue_seed{seed}/logits.npz')['assessment'].argmax(axis=1)
        task = np.load(run/f'task_full_seed{seed}/logits.npz')['assessment'].argmax(axis=1)
        raw_ok = raw == labels[ix]; task_ok = task == labels[ix]
        paired.append(dict(seed=seed, n=len(ix), raw_correct=int(raw_ok.sum()), task_correct=int(task_ok.sum()),
            raw_wrong_task_correct=int((~raw_ok & task_ok).sum()), raw_correct_task_wrong=int((raw_ok & ~task_ok).sum()),
            net_extra_correct=int(task_ok.sum()-raw_ok.sum())))
    for p, digest in fingerprints.items():
        assert hashlib.sha256((run/p).read_bytes()).hexdigest() == digest
    pd.DataFrame(rows).to_csv(out/'all_checkpoint_checks.csv', index=False)
    pd.DataFrame(paired).to_csv(out/'paired_sample_counts.csv', index=False)
    pd.DataFrame(shape_rows).to_csv(out/'task_shape_geometry.csv', index=False)
    (out/'verification.json').write_text(json.dumps({'models': len(manifest), 'split_predictions_checked': len(rows),
        'original_run_files_unchanged': True, 'original_source_commit': json.loads((run/'metadata.json').read_text())['source_commit'],
        'official_test_read': False, 'new_neural_training': False, 'shape_geometry_is_posthoc': True}, indent=2), encoding='utf-8')
    plot(run, out)


def plot(run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = pd.read_csv(run/'summary.csv'); s = summary[summary.role == 'assessment'].set_index('branch')
    order = ['raw_pretrained', 'raw_continue', 'half_continue', 'ssa_energy_continue', 'global_gate', 'task_full']
    names = ['Raw pretrained', 'Raw continued', 'Half scale', 'Fixed SSA 90%', 'Global gate', 'Task SSA']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].bar(names, s.loc[order, 'accuracy_mean']*100, yerr=s.loc[order, 'accuracy_std']*100,
                capsize=3, color=['grey', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red'])
    axes[0].set_ylim(0, 100); axes[0].tick_params(axis='x', labelrotation=30)
    for i, branch in enumerate(order):
        axes[0].text(i, s.loc[branch, 'accuracy_mean']*100+2, f'{s.loc[branch, "accuracy_mean"]*100:.2f}', ha='center', fontsize=9)
    axes[0].set_ylabel('Internal assessment accuracy (%)'); axes[0].set_title('Mean and sample SD across 3 training seeds')
    metrics = pd.read_csv(run/'metrics.csv'); a = metrics[metrics.role == 'assessment']
    wide = a[a.branch != 'majority_fit_class'].pivot(index='seed', columns='branch', values='accuracy')
    axes[1].axhline(0, color='black', linewidth=1)
    for control, label, color in [('raw_continue', 'versus raw continued', 'tab:blue'), ('global_gate', 'versus global scale', 'tab:purple')]:
        axes[1].plot(wide.index, 100*(wide.task_full-wide[control]), 'o-', label=label, color=color)
    axes[1].set_xticks([42, 43, 44]); axes[1].set_xlabel('Training seed')
    axes[1].set_ylabel('Task SSA accuracy difference (percentage points)')
    axes[1].set_title('Positive values favor task SSA'); axes[1].legend()
    fig.suptitle('FordA TRAIN-only pilot; official TEST unopened; no added corruption / SRD')
    fig.savefig(out/'classification_results.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
