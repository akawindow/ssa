"""Score replication only after every fixed seed completed; preserve original runs."""
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
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', nargs=3, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(__file__).resolve().parents[1]; jobs = [root/p for p in a.jobs]
    cfg = json.loads((jobs[0]/'protocol.json').read_text()); audit = json.loads((jobs[0]/'data_audit.json').read_text())
    done = [json.loads((p/'completed.json').read_text()) for p in jobs]
    assert sorted(d['seed'] for d in done) == cfg['seeds']
    assert all(d['training_complete'] and not d['assessment_scored'] for d in done)
    for p in jobs:
        assert json.loads((p/'protocol.json').read_text()) == cfg
        assert json.loads((p/'data_audit.json').read_text()) == audit
    sources = jobs+[root/cfg['reference_raw_run'], root/cfg['reference_normalized_run']]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for source in sources for p in source.rglob('*') if p.is_file()}
    out = root/a.out; out.mkdir(parents=True, exist_ok=False)
    datafile = root/'data/forda/FordA_TRAIN.tsv'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == audit['sha256']
    data = np.loadtxt(datafile, delimiter='\t'); y = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    with np.load(root/audit['cache']) as z:
        c = z['components']
    tf = tensorflow(101); rows = []; checks = []; pairs = []
    for job, completion in zip(jobs, done):
        seed = completion['seed']; saved_predictions = {}
        for item in json.loads((job/'manifest.json').read_text()):
            folder = job/item['folder']; variant = item['variant']
            tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
            proxy.load_weights(str(folder/'proxy.weights.h5'))
            t = json.loads((folder/'training.json').read_text())
            expected = t['initial_for_branches_sha256'] if variant == 'raw_pretrained' else t['final_proxy_sha256']
            assert weight_digest(proxy.get_weights()) == expected
            gated = variant in ['task_normalized', 'task_unnormalized']
            local = json.loads((folder/'config.json').read_text()) if variant != 'raw_pretrained' else cfg
            gate = build_gate(tf, proxy, 'task_full', local, audit['counts']['fit']) if gated else None
            if gated:
                with np.load(folder/'gate.npz') as z:
                    gate.restore([z[f'w{i}'] for i in range(len(z.files))])
                assert weight_digest(gate.weights()) == t['final_head_sha256']
            before = (weight_digest(proxy.get_weights()), weight_digest(gate.weights() if gate else []))
            inputs = c if gated else x; payload = {}
            for role, ids in audit['indices'].items():
                ix = np.array(ids); logits = predict(proxy, gate, inputs[ix])
                sc = scores(logits, y[ix]); sc.pop('confusion_matrix')
                rows.append(dict(seed=seed, role=role, variant=variant, checkpoint='best', **sc))
                if role == 'selection':
                    np.testing.assert_allclose(sc['cross_entropy'], t['best_selection_cross_entropy'], atol=1e-7)
                payload[role] = logits
                saved_predictions[(variant, role)] = logits.argmax(axis=1)
            assert before == (weight_digest(proxy.get_weights()), weight_digest(gate.weights() if gate else []))
            np.savez_compressed(out/f'{variant}_seed{seed}_logits.npz', **payload)
            checks.append(dict(seed=seed, variant=variant, weight_hash_matches=True, selection_loss_matches=True))
            # Actual epoch60 terminal states, separately from selection-best; not pretraining.
            if variant != 'raw_pretrained':
                with np.load(folder/'epoch60_last.npz') as z:
                    proxy.set_weights([z[f'proxy_w{i}'] for i in range(sum(k.startswith('proxy_') for k in z.files))])
                    if gated:
                        gate.restore([z[f'gate_w{i}'] for i in range(sum(k.startswith('gate_') for k in z.files))])
                for role in ['selection', 'assessment']:
                    ix = np.array(audit['indices'][role]); logits = predict(proxy, gate, inputs[ix])
                    sc = scores(logits, y[ix]); sc.pop('confusion_matrix')
                    rows.append(dict(seed=seed, role=role, variant=variant, checkpoint='last', **sc))
                    if role == 'selection':
                        np.testing.assert_allclose(sc['cross_entropy'], t['curve'][-1]['selection_cross_entropy'], atol=1e-7)
        for role, ids in audit['indices'].items():
            truth = y[np.array(ids)]; norm = saved_predictions[('task_normalized', role)] == truth
            for control in ['raw_continue', 'task_unnormalized']:
                ok = saved_predictions[(control, role)] == truth
                pairs.append(dict(seed=seed, role=role, control=control, n=len(truth),
                    normalized_correct=int(norm.sum()), control_correct=int(ok.sum()),
                    control_wrong_normalized_correct=int((~ok & norm).sum()),
                    control_correct_normalized_wrong=int((ok & ~norm).sum()),
                    net_extra_correct=int(norm.sum()-ok.sum()), accuracy_difference_pp=float(100*(norm.mean()-ok.mean()))))
        print('Verified best/last replication models seed', seed, flush=True)
    new = pd.DataFrame(rows); new.to_csv(out/'new_metrics.csv', index=False)
    old = pd.read_csv(root/cfg['reference_raw_run']/'metrics.csv')
    old = old[old.branch.isin(['raw_continue', 'task_full'])].rename(columns={'branch': 'variant'})
    old.loc[old.variant == 'task_full', 'variant'] = 'task_unnormalized'
    oldnorm = pd.read_csv(root/cfg['reference_normalized_run']/'metrics.csv')
    oldnorm = oldnorm[oldnorm.branch == 'task_full'].rename(columns={'branch': 'variant'})
    oldnorm['variant'] = 'task_normalized'; old = pd.concat([old, oldnorm], ignore_index=True)
    old['checkpoint'] = 'best'; old['seed'] = old.seed.astype(int)
    both = pd.concat([old, new[new.checkpoint == 'best']], ignore_index=True)
    both.to_csv(out/'all_six_metrics.csv', index=False)
    summaries = []
    for cohort, f in [('old_three', old), ('new_three', new[new.checkpoint == 'best']), ('all_six', both)]:
        s = f.groupby(['role', 'variant'])[['accuracy', 'macro_f1', 'cross_entropy']].agg(['mean', 'std'])
        s.columns = ['_'.join(k) for k in s.columns]; s = s.reset_index(); s['cohort'] = cohort; summaries.append(s)
    pd.concat(summaries).to_csv(out/'cohort_summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(out/'new_paired_differences.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'model_checks.csv', index=False)
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (out/'verification.json').write_text(json.dumps(dict(new_common_pretraining_models=3,
        new_branches=9, selected_models_reloaded=len(checks), split_scores=len(rows),
        all_training_completed_before_scoring=True, source_files_checked=len(hashes), source_files_unchanged=True,
        all_selection_losses_and_weights_match=True, official_test_read=False, new_training_during_scoring=False), indent=2), encoding='utf-8')
    print(pd.concat(summaries).query("role == 'assessment'").to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).query("role == 'assessment'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
