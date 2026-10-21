"""Frozen task classifier: learned reconstruction vs its per-sample raw projection."""
import os
for key in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.classification import build_classifier, build_gate, predict, scores
from ssa_task_srd.learning import tensorflow, weight_digest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='configs/classification_scale_diagnostic.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/args.protocol).read_text(encoding='utf-8'))
    run = root/protocol['source_run']; out = root/args.out
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    fingerprints = {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()}
    cfg = json.loads((run/'protocol.json').read_text())
    audit = json.loads((run/'data_audit.json').read_text())
    dataset = root/'data/forda/FordA_TRAIN.tsv'
    assert digest(dataset) == audit['sha256']
    data = np.loadtxt(dataset, delimiter='\t')
    y = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    with np.load(root/audit['cache']) as cache:
        components = cache['components']
    np.testing.assert_allclose(components.sum(axis=1), x, atol=2e-6, rtol=2e-6)
    rows = []; pairs = []; geometry = []; checks = []; sample_rows = []
    tf = tensorflow(101)
    for seed in protocol['seeds']:
        tf.keras.backend.clear_session()
        folder = run/f'task_full_seed{seed}'
        proxy = build_classifier(tf, cfg['length'])
        proxy.load_weights(str(folder/'proxy.weights.h5'))
        gate = build_gate(tf, proxy, 'task_full', cfg, audit['counts']['fit'])
        with np.load(folder/'gate.npz') as z:
            gate.restore([z[f'w{i}'] for i in range(len(z.files))])
        training = json.loads((folder/'training.json').read_text())
        proxy_before = weight_digest(proxy.get_weights()); head_before = weight_digest(gate.weights())
        assert proxy_before == training['final_proxy_sha256']
        assert head_before == training['final_head_sha256']
        proxy.trainable = False; gate.head.trainable = False
        saved = np.load(folder/'logits.npz')
        arrays = {}
        for role in protocol['roles']:
            ix = np.array(audit['indices'][role]); raw = x[ix].astype(np.float64)
            rebuilt = np.concatenate([gate.reconstruct(components[ix[s:s+cfg['batch_size']]])[0].numpy()
                for s in range(0, len(ix), cfg['batch_size'])])
            rebuilt64 = rebuilt.astype(np.float64)
            denominator = np.square(raw).sum(axis=1)
            scale = np.divide((raw*rebuilt64).sum(axis=1), denominator,
                out=np.zeros(len(raw)), where=denominator > 0)
            scaled64 = scale[:, None]*raw
            residual = rebuilt64-scaled64
            orthogonality = np.abs((raw*residual).sum(axis=1))
            assert np.max(orthogonality) < 1e-9
            # Unconstrained least-squares projection must beat the neighboring scalar choices.
            for offset in [-.01, .01]:
                assert np.all(np.square(residual).sum(axis=1) <=
                    np.square(rebuilt64-(scale[:, None]+offset)*raw).sum(axis=1)+1e-10)
            logits = {}
            for variant, values in [('task_reconstruction', rebuilt), ('closest_sample_scale', scaled64.astype(np.float32))]:
                z = predict(proxy, None, values, cfg['batch_size'])
                logits[variant] = z
                sc = scores(z, y[ix]); cm = sc.pop('confusion_matrix')
                rows.append(dict(seed=seed, role=role, variant=variant,
                    selected_epoch=training['best_epoch'], **sc, confusion_matrix=json.dumps(cm)))
                arrays[f'{role}_{variant}_logits'] = z
            np.testing.assert_allclose(logits['task_reconstruction'], saved[role], atol=2e-6, rtol=2e-6)
            if role == 'selection':
                np.testing.assert_allclose(rows[-2]['cross_entropy'], training['best_selection_cross_entropy'], atol=1e-7)
            task_pred = logits['task_reconstruction'].argmax(axis=1)
            scale_pred = logits['closest_sample_scale'].argmax(axis=1)
            task_ok = task_pred == y[ix]; scale_ok = scale_pred == y[ix]
            pairs.append(dict(seed=seed, role=role, n=len(ix), task_correct=int(task_ok.sum()),
                scale_correct=int(scale_ok.sum()), scale_wrong_task_correct=int((~scale_ok & task_ok).sum()),
                scale_correct_task_wrong=int((scale_ok & ~task_ok).sum()),
                prediction_disagreements=int((task_pred != scale_pred).sum()),
                net_extra_correct=int(task_ok.sum()-scale_ok.sum()),
                task_minus_scale_accuracy_pp=float((task_ok.mean()-scale_ok.mean())*100),
                task_minus_scale_ce=rows[-2]['cross_entropy']-rows[-1]['cross_entropy']))
            geometry.append(dict(seed=seed, role=role, scale_mean=float(scale.mean()),
                scale_min=float(scale.min()), scale_max=float(scale.max()),
                shape_rms_over_raw_rms=float(np.sqrt(np.square(residual).mean()/np.square(raw).mean())),
                max_projection_inner_product=float(orthogonality.max())))
            for j, sample in enumerate(ix):
                sample_rows.append(dict(seed=seed, role=role, source_row=int(sample), label=int(y[sample]),
                    scale=float(scale[j]), task_prediction=int(task_pred[j]), scale_prediction=int(scale_pred[j])))
            arrays[f'{role}_scale'] = scale
            checks.append(dict(seed=seed, role=role, task_predictions_reproduced=True,
                projection_orthogonal=True, projection_minimum_checked=True))
        assert weight_digest(proxy.get_weights()) == proxy_before
        assert weight_digest(gate.weights()) == head_before
        np.savez_compressed(out/f'predictions_seed{seed}.npz', **arrays)
        print('Verified frozen substitutions seed', seed, flush=True)
    assert all(digest(run/p) == h for p, h in fingerprints.items())
    f = pd.DataFrame(rows)
    f.to_csv(out/'metrics.csv', index=False)
    summary = f.groupby(['role', 'variant'])[['accuracy', 'balanced_accuracy', 'macro_f1', 'cross_entropy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')
    pd.DataFrame(pairs).to_csv(out/'paired_differences.csv', index=False)
    pd.DataFrame(geometry).to_csv(out/'geometry.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'checks.csv', index=False)
    pd.DataFrame(sample_rows).to_csv(out/'sample_predictions.csv', index=False)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(dict(frozen_models=3, split_pairs=len(checks),
        split_scores=len(rows), all_source_predictions_reproduced=True, all_model_weights_unchanged=True,
        source_files_checked=len(fingerprints), all_source_files_unchanged=True,
        official_test_read=False, new_training=False), indent=2), encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(dict(source_commit=subprocess.check_output(
        ['D:/program/Git/cmd/git.exe', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_run_commit=json.loads((run/'metadata.json').read_text())['source_commit'],
        elapsed_seconds=time.perf_counter()-start, dataset_sha256=audit['sha256']), indent=2), encoding='utf-8')
    print(summary.loc['assessment'].to_string(), flush=True)
    print(pd.DataFrame(pairs).query("role == 'assessment'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
