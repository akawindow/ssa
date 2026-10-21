"""Reload selected and terminal budget states of the selector-only normalization run."""
import os
for k in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(k, '1')
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
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(__file__).resolve().parents[1]; run = root/a.run; out = root/a.out
    out.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    cfg = json.loads((run/'protocol.json').read_text()); audit = json.loads((run/'data_audit.json').read_text())
    data = np.loadtxt(root/'data/forda/FordA_TRAIN.tsv', delimiter='\t'); y = (data[:, 0] == 1).astype(np.int64)
    with np.load(root/audit['cache']) as z:
        c = z['components']
    rows = []; states = 0; tf = tensorflow(101)
    for seed in cfg['seeds']:
        folder = run/f'task_full_seed{seed}'; t = json.loads((folder/'training.json').read_text())
        tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
        gate = build_gate(tf, proxy, 'task_full', cfg, audit['counts']['fit'])
        assert t['classifier_optimizer_steps'] == t['head_optimizer_steps'] == 2400
        for budget in [40, 50, 60]:
            record = next(r for r in t['snapshots'] if r['epoch'] == budget)
            for kind in ['best', 'last']:
                with np.load(folder/f'epoch{budget}_{kind}.npz') as z:
                    proxy.set_weights([z[f'proxy_w{i}'] for i in range(sum(k.startswith('proxy_') for k in z.files))])
                    gate.restore([z[f'gate_w{i}'] for i in range(sum(k.startswith('gate_') for k in z.files))])
                if kind == 'best':
                    assert weight_digest(proxy.get_weights()) == record['best_proxy_sha256']
                    assert weight_digest(gate.weights()) == record['best_head_sha256']
                for role in ['selection', 'assessment']:
                    ix = np.array(audit['indices'][role]); logits = predict(proxy, gate, c[ix])
                    sc = scores(logits, y[ix]); sc.pop('confusion_matrix')
                    if role == 'selection':
                        expected = record['best_selection_cross_entropy'] if kind == 'best' else record['last_selection_cross_entropy']
                        np.testing.assert_allclose(sc['cross_entropy'], expected, atol=1e-7)
                    if budget == 60 and kind == 'best':
                        np.testing.assert_allclose(logits, np.load(folder/'logits.npz')[role], atol=2e-6, rtol=2e-6)
                    rows.append(dict(seed=seed, budget=budget, checkpoint=kind, role=role,
                        selected_epoch=record['best_epoch'] if kind == 'best' else budget, **sc))
                states += 1
        print('Verified normalized budget checkpoints seed', seed, flush=True)
    f = pd.DataFrame(rows); f.to_csv(out/'checkpoint_metrics.csv', index=False)
    s = f.groupby(['role', 'budget', 'checkpoint'])[['accuracy', 'macro_f1', 'cross_entropy']].agg(['mean', 'std'])
    s.columns = ['_'.join(k) for k in s.columns]; s.to_csv(out/'checkpoint_summary.csv')
    assert all(hashlib.sha256((run/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (out/'verification.json').write_text(json.dumps(dict(states_checked=states, split_scores=len(f),
        all_selection_losses_match=True, final_best_predictions_match=True, source_files_unchanged=True,
        official_test_read=False, new_training=False), indent=2), encoding='utf-8')
    print(s.loc['assessment'].to_string(), flush=True)


if __name__ == '__main__':
    main()
