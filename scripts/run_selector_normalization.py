"""One-factor experiment; reuse immutable matched controls, train only new selectors."""
import os
for k in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(k, '1')
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
from ssa_task_srd.classification import build_classifier, build_gate, train_branch, predict, scores
from ssa_task_srd.learning import tensorflow, weight_digest


def write(p, v):
    p.write_text(json.dumps(v, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', default='configs/classification_selector_rms_protocol.json')
    args = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/args.protocol).read_text()); out = root/args.out
    out.mkdir(parents=True, exist_ok=False); tick = time.perf_counter()
    source = root/cfg['baseline_from']; old_cfg = json.loads((source/'protocol.json').read_text())
    for k, v in old_cfg.items():
        if k not in ['prefix_from', 'scope', 'budget', 'stop_rule']:
            assert cfg[k] == v, f'Unexpected change: {k}'
    fingerprints = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob('*') if p.is_file()}
    audit = json.loads((source/'data_audit.json').read_text())
    datafile = root/'data/forda/FordA_TRAIN.tsv'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == audit['sha256']
    data = np.loadtxt(datafile, delimiter='\t'); y = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    with np.load(root/audit['cache']) as z:
        c = z['components']
    np.testing.assert_allclose(c.sum(axis=1), x, atol=2e-6, rtol=2e-6)
    split = {k: np.array(v) for k, v in audit['indices'].items()}
    write(out/'protocol.json', cfg); write(out/'data_audit.json', audit)
    manifest = []; audits = []
    for seed in cfg['seeds']:
        tf = tensorflow(seed); tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
        proxy.load_weights(str(source/f'raw_pretrained_seed{seed}/proxy.weights.h5'))
        old = json.loads((source/f'task_full_seed{seed}/training.json').read_text())
        assert weight_digest(proxy.get_weights()) == old['initial_proxy_sha256']
        folder = out/f'task_full_seed{seed}'; folder.mkdir(); start = time.perf_counter()
        def snapshot(record, bp, bg, lp, lg):
            for kind, pw, gw in [('best', bp, bg), ('last', lp, lg)]:
                if kind == 'last' and record['epoch'] not in cfg['terminal_epochs']:
                    continue
                payload = {f'proxy_w{i}': w for i, w in enumerate(pw)}
                payload.update({f'gate_w{i}': w for i, w in enumerate(gw)})
                np.savez_compressed(folder/f'epoch{record["epoch"]}_{kind}.npz', **payload)
        print('Training normalized selector seed', seed, flush=True)
        gate, t = train_branch(tf, proxy, 'task_full', c[split['fit']], y[split['fit']],
            c[split['selection']], y[split['selection']], cfg, seed, snapshot)
        for key in ['initial_proxy_sha256', 'batch_order_sha256', 'classifier_optimizer_steps',
                    'head_optimizer_steps', 'classifier_parameters', 'head_parameters']:
            assert t[key] == old[key], key
        proxy.save_weights(str(folder/'proxy.weights.h5'))
        np.savez_compressed(folder/'gate.npz', **{f'w{i}': w for i, w in enumerate(gate.weights())})
        t['elapsed_seconds'] = time.perf_counter()-start; t['matched_to_source_control'] = True
        write(folder/'training.json', t); audits.append(t)
        manifest.append(dict(seed=seed, branch='task_full', folder=folder.name))
    write(out/'manifest.json', manifest); write(out/'training_audit.json', {'runs': audits})
    # No assessment scoring before the entire prespecified training is complete.
    rows = []; checks = []; geometry = []; weights = []
    for item in manifest:
        seed = item['seed']; folder = out/item['folder']; tf.keras.backend.clear_session()
        proxy = build_classifier(tf, cfg['length']); proxy.load_weights(str(folder/'proxy.weights.h5'))
        gate = build_gate(tf, proxy, 'task_full', cfg, len(split['fit']))
        with np.load(folder/'gate.npz') as z:
            gate.restore([z[f'w{i}'] for i in range(len(z.files))])
        t = json.loads((folder/'training.json').read_text()); predictions = {}
        assert weight_digest(proxy.get_weights()) == t['final_proxy_sha256']
        assert weight_digest(gate.weights()) == t['final_head_sha256']
        for role, ix in split.items():
            logits = predict(proxy, gate, c[ix]); predictions[role] = logits
            sc = scores(logits, y[ix]); cm = sc.pop('confusion_matrix')
            rows.append(dict(seed=seed, branch='task_full', role=role, **sc, confusion_matrix=json.dumps(cm)))
            if role == 'selection':
                np.testing.assert_allclose(sc['cross_entropy'], t['best_selection_cross_entropy'], atol=1e-7)
            rebuilt = []; ww = []
            for s in range(0, len(ix), 64):
                z, w = gate.reconstruct(c[ix[s:s+64]]); rebuilt.append(z.numpy()); ww.append(w.numpy()[:, :, 0])
            rebuilt = np.concatenate(rebuilt).astype(np.float64); ww = np.concatenate(ww)
            raw = x[ix].astype(np.float64); scale = (raw*rebuilt).sum(axis=1)/(raw*raw).sum(axis=1)
            geometry.append(dict(seed=seed, role=role, closest_scale_mean=float(scale.mean()),
                shape_rms_over_raw_rms=float(np.sqrt(np.square(rebuilt-scale[:, None]*raw).mean()/np.square(raw).mean())),
                within_sample_weight_std=float(ww.std(axis=1).mean())))
            for j in range(c.shape[1]):
                weights.append(dict(seed=seed, role=role, component=j+1, retention_mean=float(ww[:, j].mean()),
                    retention_std=float(ww[:, j].std())))
            checks.append(dict(seed=seed, role=role, weights_reloaded=True, selection_loss_verified=True))
        np.savez_compressed(folder/'logits.npz', **predictions)
    frame = pd.DataFrame(rows)
    baseline = pd.read_csv(source/'metrics.csv')
    baseline.loc[baseline.branch == 'task_full', 'branch'] = 'task_unnormalized'
    frame = pd.concat([baseline, frame], ignore_index=True); frame.to_csv(out/'metrics.csv', index=False)
    summary = frame.groupby(['role', 'branch'])[['accuracy', 'balanced_accuracy', 'macro_f1', 'cross_entropy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(k) for k in summary.columns]; summary.to_csv(out/'summary.csv')
    pd.DataFrame(geometry).to_csv(out/'geometry.csv', index=False)
    pd.DataFrame(weights).to_csv(out/'component_weights.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'reload_checks.csv', index=False)
    assert all(hashlib.sha256((source/p).read_bytes()).hexdigest() == h for p, h in fingerprints.items())
    write(out/'verification.json', dict(source_files_checked=len(fingerprints), source_files_unchanged=True,
        same_initialization_batches_steps=True, completed_new_models=3, all_phase_audits_passed=True,
        assessment_not_used_for_training=True, official_test_read=False))
    write(out/'metadata.json', dict(source_commit=subprocess.check_output(['D:/program/Git/cmd/git.exe', 'rev-parse', 'HEAD'],
        cwd=root, text=True).strip(), elapsed_seconds=time.perf_counter()-tick,
        source_control_commit=json.loads((source/'metadata.json').read_text())['source_commit'],
        scope=cfg['scope'], reused_controls=True))
    print(summary.loc['assessment'].to_string(), flush=True)


if __name__ == '__main__':
    main()
