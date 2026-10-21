"""Prespecified posthoc weight, gradient and frozen-feature linear-probe audit."""
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
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ssa_task_srd.classification import build_classifier, build_gate, predict, scores, ce
from ssa_task_srd.learning import tensorflow, weight_digest


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', default='configs/classification_component_audit.json')
    args = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root/args.protocol).read_text())
    out = root/args.out; out.mkdir(parents=True, exist_ok=False); tick = time.perf_counter()
    run = root/protocol['source_run']; cfg = json.loads((run/'protocol.json').read_text())
    audit = json.loads((run/'data_audit.json').read_text())
    fingerprints = {str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()}
    path = root/'data/forda/FordA_TRAIN.tsv'; assert sha(path) == audit['sha256']
    data = np.loadtxt(path, delimiter='\t'); y = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    with np.load(root/audit['cache']) as z:
        c = z['components']
    split = {k: np.array(v) for k, v in audit['indices'].items()}
    rows = []; probe_rows = []; mech = []; correlations = []
    tf = tensorflow(101)
    for seed in protocol['seeds']:
        tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
        folder = run/f'task_full_seed{seed}'; training = json.loads((folder/'training.json').read_text())
        proxy.load_weights(str(folder/'proxy.weights.h5'))
        gate = build_gate(tf, proxy, 'task_full', cfg, audit['counts']['fit'])
        with np.load(folder/'gate.npz') as z:
            gate.restore([z[f'w{i}'] for i in range(len(z.files))])
        ph = weight_digest(proxy.get_weights()); gh = weight_digest(gate.weights())
        assert ph == training['final_proxy_sha256'] and gh == training['final_head_sha256']
        initial = build_classifier(tf, cfg['length'])
        initial.load_weights(str(run/f'raw_pretrained_seed{seed}/proxy.weights.h5'))
        initial_head = build_gate(tf, initial, 'task_full', cfg, audit['counts']['fit'])
        pw0 = initial.get_weights(); hw0 = initial_head.weights()
        assert weight_digest(pw0) == training['initial_proxy_sha256']
        # CE gradients only: not an optimizer update or new backbone training.
        ix = split['fit'][:64]
        with tf.GradientTape() as tape:
            rebuilt, _ = gate.reconstruct(c[ix])
            loss = ce(tf, y[ix], proxy(rebuilt, training=False))
        grads = tape.gradient(loss, gate.parameters())
        assert all(g is not None for g in grads)
        gradient_norm = float(tf.linalg.global_norm(grads).numpy()); assert gradient_norm > 0
        features = []
        for start in range(0, len(c), 64):
            part = c[start:start+64]
            f = proxy.layers[0](part.reshape(-1, cfg['length']), training=False).numpy()
            features.append(f.reshape(len(part), c.shape[1], 32))
        features = np.concatenate(features)
        kernel, bias = gate.weights(); preactivation = features @ kernel[:, 0] + bias[0]
        weights = expit(preactivation).astype(np.float32)
        rebuilt = np.sum(c*weights[:, :, None], axis=1)
        saved = np.load(folder/'logits.npz')
        for role, ix in split.items():
            logits = predict(proxy, None, rebuilt[ix])
            np.testing.assert_allclose(logits, saved[role], atol=3e-6, rtol=3e-6)
            for j in range(c.shape[1]):
                rows.append(dict(seed=seed, role=role, component=j+1,
                    residual=(j == 32), retention_mean=float(weights[ix, j].mean()),
                    retention_std=float(weights[ix, j].std()),
                    component_rms=float(np.sqrt(np.square(c[ix, j].astype(np.float64)).mean())),
                    logit_feature_mean=float((preactivation[ix, j]-bias[0]).mean())))
        raw_features = np.concatenate([proxy.layers[0](x[s:s+64], training=False).numpy() for s in range(0, len(x), 64)])
        for component in range(33):
            f = raw_features if component == 32 else features[:, component]
            name = 'raw' if component == 32 else f'C{component+1}'
            scaler = StandardScaler().fit(f[split['fit']])
            transformed = scaler.transform(f)
            probe = LogisticRegression(C=1., solver='lbfgs', max_iter=2000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                probe.fit(transformed[split['fit']], y[split['fit']])
            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            for role, ix in split.items():
                margin = probe.decision_function(transformed[ix])
                sc = scores(np.column_stack([np.zeros(len(ix)), margin]), y[ix]); sc.pop('confusion_matrix')
                probe_rows.append(dict(seed=seed, role=role, component=name, **sc))
        assessment = pd.DataFrame(probe_rows)
        a = assessment[(assessment.seed == seed) & (assessment.role == 'assessment')].set_index('component')
        fit_weights = weights[split['fit'], :32].mean(axis=0)
        fit_rms = np.sqrt(np.square(c[split['fit'], :32].astype(np.float64)).mean(axis=(0, 2)))
        corr = spearmanr(fit_weights, [a.loc[f'C{j+1}', 'accuracy'] for j in range(32)]).statistic
        correlations.append(dict(seed=seed, weight_vs_probe_accuracy_spearman=float(corr),
            weight_vs_rms_spearman=float(spearmanr(fit_weights, fit_rms).statistic)))
        encoder_delta = np.sqrt(sum(np.square(a.astype(np.float64)-b).sum()
            for a, b in zip(initial.layers[0].get_weights(), proxy.layers[0].get_weights())))
        assert encoder_delta > 0 and not np.array_equal(hw0[0], kernel)
        assert training['classifier_optimizer_steps'] == training['head_optimizer_steps'] == 2400
        assert all(training['first_batch_audit'].values())
        mech.append(dict(seed=seed, selected_epoch=training['best_epoch'],
            classifier_steps=training['classifier_optimizer_steps'], head_steps=training['head_optimizer_steps'],
            phase_isolation_passed=True, encoder_initial_to_selected_l2=float(encoder_delta),
            head_initial_to_selected_l2=float(np.sqrt(sum(np.square(a.astype(np.float64)-b).sum() for a, b in zip(hw0, gate.weights())))),
            selector_ce_gradient_norm=gradient_norm, head_kernel_norm=float(np.linalg.norm(kernel)),
            head_bias=float(bias[0]), bias_only_retention=float(expit(bias[0])),
            assessment_mean_retention=float(weights[split['assessment']].mean()),
            assessment_within_sample_retention_std_mean=float(weights[split['assessment']].std(axis=1).mean()),
            assessment_within_sample_logit_std_mean=float(preactivation[split['assessment']].std(axis=1).mean())))
        assert ph == weight_digest(proxy.get_weights()) and gh == weight_digest(gate.weights())
        np.savez_compressed(out/f'features_weights_seed{seed}.npz', features=features, weights=weights)
        print('Verified mechanism and 33 linear probes seed', seed, flush=True)
    assert all(sha(run/p) == h for p, h in fingerprints.items())
    for name, values in [('component_weights', rows), ('component_probes', probe_rows),
        ('mechanism_audit', mech), ('correlations', correlations)]:
        pd.DataFrame(values).to_csv(out/f'{name}.csv', index=False)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(dict(source_files_checked=len(fingerprints),
        source_files_unchanged=True, original_task_predictions_reproduced=True, all_model_weights_unchanged=True,
        actual_phase_audits_passed=True, nonzero_head_ce_gradients=True, linear_probes=99,
        linear_probe_split_scores=len(probe_rows), new_backbone_training=False, official_test_read=False,
        elapsed_seconds=time.perf_counter()-tick, source_commit=subprocess.check_output(
            ['D:/program/Git/cmd/git.exe', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()), indent=2), encoding='utf-8')
    print(pd.DataFrame(mech).to_string(index=False), flush=True)
    print(pd.DataFrame(correlations).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
