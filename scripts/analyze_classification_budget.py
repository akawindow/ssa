"""Prespecified budget checkpoints, terminal states, and source-prefix verification."""
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
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    run = root/args.run; out = root/args.out; out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((run/'protocol.json').read_text()); audit = json.loads((run/'data_audit.json').read_text())
    source = root/cfg['pretrained_from']
    final_epoch = cfg['collaborative_epochs']
    terminal_epochs = cfg.get('terminal_epochs', [final_epoch])
    fingerprints = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    data = np.loadtxt(root/'data/forda/FordA_TRAIN.tsv', delimiter='\t')
    labels = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    cache = np.load(root/audit['cache']); components = cache['components']; fixed = cache['fixed']
    inputs = {'raw_continue': x, 'half_continue': .5*x, 'ssa_energy_continue': fixed, 'global_gate': x, 'task_full': components}
    rows = []; checks = []; tf = tensorflow(101)
    for seed in cfg['seeds']:
        for branch in cfg['branches']:
            folder = run/f'{branch}_seed{seed}'; training = json.loads((folder/'training.json').read_text())
            steps = final_epoch*int(np.ceil(audit['counts']['fit']/cfg['batch_size']))
            assert training['classifier_optimizer_steps'] == steps and len(training['curve']) == final_epoch+1
            assert training['snapshots'][0]['old_ten_epoch_best_exactly_reproduced']
            tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
            gate = build_gate(tf, proxy, branch, cfg, audit['counts']['fit']) if branch in ['task_full', 'global_gate'] else None
            snapshots = [(epoch, 'best') for epoch in cfg['snapshot_epochs']]+[(e, 'last') for e in terminal_epochs]
            for budget, kind in snapshots:
                with np.load(folder/f'epoch{budget}_{kind}.npz') as z:
                    pw = [z[f'proxy_w{i}'] for i in range(sum(k.startswith('proxy_') for k in z.files))]
                    gw = [z[f'gate_w{i}'] for i in range(sum(k.startswith('gate_') for k in z.files))]
                proxy.set_weights(pw)
                if gate:
                    gate.restore(gw)
                record = next(r for r in training['snapshots'] if r['epoch'] == budget)
                selected = record['best_epoch'] if kind == 'best' else budget
                if kind == 'best':
                    assert weight_digest(proxy.get_weights()) == record['best_proxy_sha256']
                    assert weight_digest(gate.weights() if gate else []) == record['best_head_sha256']
                roles = ['selection', 'assessment']+(['fit'] if budget == final_epoch else [])
                for role in roles:
                    ix = np.array(audit['indices'][role]); logits = predict(proxy, gate, inputs[branch][ix])
                    sc = scores(logits, labels[ix]); cm = sc.pop('confusion_matrix')
                    rows.append(dict(seed=seed, branch=branch, budget=budget, checkpoint=kind,
                        selected_epoch=selected, role=role, **sc, confusion_matrix=json.dumps(cm)))
                    if role == 'selection':
                        expected = record['best_selection_cross_entropy'] if kind == 'best' else record['last_selection_cross_entropy']
                        np.testing.assert_allclose(sc['cross_entropy'], expected, atol=1e-7)
                    if budget == 10 and kind == 'best':
                        saved = np.load(source/f'{branch}_seed{seed}/logits.npz')[role]
                        np.testing.assert_allclose(logits, saved, atol=2e-6, rtol=2e-6)
                    if budget == final_epoch and kind == 'best':
                        saved = np.load(folder/'logits.npz')[role]
                        np.testing.assert_allclose(logits, saved, atol=2e-6, rtol=2e-6)
                    if cfg.get('prefix_from') and budget == 40 and kind == 'best':
                        saved = np.load(root/cfg['prefix_from']/f'{branch}_seed{seed}/logits.npz')[role]
                        np.testing.assert_allclose(logits, saved, atol=2e-6, rtol=2e-6)
                        assert record['old_prefix_best_and_terminal_exactly_reproduced']
                checks.append(dict(seed=seed, branch=branch, budget=budget, checkpoint=kind,
                    selection_loss_matches=True, selected_weights_reloaded=True,
                    ten_epoch_predictions_match_old_run=True if budget == 10 else None))
            print('Verified budget checkpoints', branch, seed, flush=True)
    f = pd.DataFrame(rows); f.to_csv(out/'checkpoint_metrics.csv', index=False)
    summary = f.groupby(['role', 'budget', 'checkpoint', 'branch'])[['accuracy', 'macro_f1', 'cross_entropy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]; summary.to_csv(out/'checkpoint_summary.csv')
    pairs = []
    for (role, budget, kind), group in f.groupby(['role', 'budget', 'checkpoint']):
        wide = group.pivot(index='seed', columns='branch', values='accuracy')
        for control in ['raw_continue', 'half_continue', 'ssa_energy_continue', 'global_gate']:
            for seed, delta in (wide.task_full-wide[control]).items():
                pairs.append(dict(role=role, budget=budget, checkpoint=kind, control=control, seed=seed,
                    task_minus_control_accuracy_pp=float(delta*100)))
    pd.DataFrame(pairs).to_csv(out/'checkpoint_paired_differences.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'checkpoint_checks.csv', index=False)
    assert all(hashlib.sha256((run/p).read_bytes()).hexdigest() == h for p, h in fingerprints.items())
    (out/'verification.json').write_text(json.dumps({'checkpoint_states': len(checks),
        'split_scores_verified': len(f), 'all_15_old_ten_epoch_prefixes_reproduced': True,
        'final_budget': final_epoch, 'all_final_best_predictions_match_primary_run': True,
        'old_40_epoch_predictions_reproduced': bool(cfg.get('prefix_from')), 'source_run_files_unchanged': True,
        'new_neural_training': False, 'official_test_read': False}, indent=2), encoding='utf-8')
    plot(out)
    print(summary.loc['assessment'].to_string(), flush=True)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = pd.read_csv(out/'checkpoint_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for branch, label in [('raw_continue', 'Raw continued'), ('half_continue', 'Half scale'),
        ('ssa_energy_continue', 'Fixed SSA 90%'), ('global_gate', 'Global gate'), ('task_full', 'Task SSA')]:
        for ax, role, metric in [(axes[0], 'selection', 'cross_entropy'), (axes[1], 'assessment', 'accuracy')]:
            a = f[(f.branch == branch) & (f.role == role) & (f.checkpoint == 'best')]
            s = a.groupby('budget')[metric].agg(['mean', 'std'])
            scale = 100 if metric == 'accuracy' else 1
            ax.errorbar(s.index, s['mean']*scale, yerr=s['std']*scale, marker='o', capsize=3, label=label)
            ax.set_xticks(sorted(f.budget.unique())); ax.set_xlabel('Executed collaborative training epochs')
    axes[0].set_ylabel('Selection cross-entropy (lower is better)')
    axes[1].set_ylabel('Explored assessment accuracy (%)')
    axes[0].set_title('Checkpoint selection uses this split only'); axes[0].legend(fontsize=8)
    axes[1].set_title('Assessment never selects epochs or branches')
    fig.suptitle('Same pretrained weights; original first 10 epochs reproduced exactly; 3 seeds')
    fig.savefig(out/'budget_results.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
