"""One fixed replication seed: fresh common pretrain and 3 matched branches; no assessment scoring."""
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
from ssa_task_srd.classification import build_classifier, pretrain, train_branch
from ssa_task_srd.learning import tensorflow, weight_digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', default='configs/classification_seed_confirmation.json')
    a = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/a.protocol).read_text()); assert a.seed in cfg['seeds']
    out = root/a.out; out.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    references = [root/cfg['reference_raw_run'], root/cfg['reference_normalized_run']]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for reference in references for p in reference.rglob('*') if p.is_file()}
    prior = json.loads((references[1]/'protocol.json').read_text())
    for key in ['dataset', 'train_sha256', 'split_seed', 'length', 'embedding', 'energy_threshold',
                'batch_size', 'pretrain_epochs', 'pretrain_patience', 'collaborative_epochs',
                'learning_rate', 'selector_bias', 'selector_max_norm', 'schedule_epochs',
                'selector_rms_floor', 'backbone', 'checkpoint', 'normalization_rule']:
        assert cfg[key] == prior[key], key
    audit = json.loads((references[0]/'data_audit.json').read_text())
    dataset = root/'data/forda/FordA_TRAIN.tsv'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == audit['sha256']
    data = np.loadtxt(dataset, delimiter='\t'); y = (data[:, 0] == 1).astype(np.int64)
    x = ((data[:, 1:]-audit['scale_mean'])/audit['scale_std']).astype(np.float32)
    with np.load(root/audit['cache']) as z:
        c = z['components']
    np.testing.assert_allclose(c.sum(axis=1), x, atol=2e-6, rtol=2e-6)
    split = {k: np.array(v) for k, v in audit['indices'].items()}
    write(out/'protocol.json', cfg); write(out/'data_audit.json', audit)
    tf = tensorflow(a.seed); tf.keras.backend.clear_session()
    proxy = build_classifier(tf, cfg['length'])
    print('Fresh common pretraining seed', a.seed, flush=True)
    t = pretrain(tf, proxy, x[split['fit']], y[split['fit']], x[split['selection']], y[split['selection']], cfg, a.seed)
    initial = proxy.get_weights(); initial_hash = weight_digest(initial)
    folder = out/'raw_pretrained'; folder.mkdir(); proxy.save_weights(str(folder/'proxy.weights.h5'))
    t['initial_for_branches_sha256'] = initial_hash; write(folder/'training.json', t)
    manifest = [dict(seed=a.seed, variant='raw_pretrained', folder=folder.name, core_branch='raw_continue')]
    audits = []
    for variant in cfg['variants']:
        tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length']); proxy.set_weights(initial)
        core_branch = 'raw_continue' if variant == 'raw_continue' else 'task_full'
        local = dict(cfg, selector_input_normalization='component_rms' if variant == 'task_normalized' else 'none')
        inputs = x if core_branch == 'raw_continue' else c
        folder = out/variant; folder.mkdir(); write(folder/'config.json', local)
        def snapshot(record, bp, bg, lp, lg):
            for kind, pw, gw in [('best', bp, bg), ('last', lp, lg)]:
                payload = {f'proxy_w{i}': v for i, v in enumerate(pw)}
                payload.update({f'gate_w{i}': v for i, v in enumerate(gw)})
                np.savez_compressed(folder/f'epoch60_{kind}.npz', **payload)
        print('Training seed', a.seed, 'variant', variant, flush=True); tick = time.perf_counter()
        gate, t = train_branch(tf, proxy, core_branch, inputs[split['fit']], y[split['fit']],
            inputs[split['selection']], y[split['selection']], local, a.seed, snapshot)
        assert t['initial_proxy_sha256'] == initial_hash
        proxy.save_weights(str(folder/'proxy.weights.h5'))
        if gate:
            np.savez_compressed(folder/'gate.npz', **{f'w{i}': v for i, v in enumerate(gate.weights())})
        t.update(variant=variant, elapsed_seconds=time.perf_counter()-tick)
        write(folder/'training.json', t); audits.append(t)
        manifest.append(dict(seed=a.seed, variant=variant, folder=folder.name, core_branch=core_branch))
    for key in ['initial_proxy_sha256', 'batch_order_sha256', 'classifier_optimizer_steps', 'classifier_parameters']:
        assert len({t[key] for t in audits}) == 1, key
    assert all(t['classifier_optimizer_steps'] == 2400 for t in audits)
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    write(out/'manifest.json', manifest); write(out/'training_audit.json', {'runs': audits})
    write(out/'completed.json', dict(seed=a.seed, training_complete=True, common_pretrain_is_fresh=True,
        all_variants_complete=True, all_budgets_batches_initializations_match=True,
        assessment_scored=False, official_test_read=False, references_unchanged=True,
        reference_files_checked=len(hashes), elapsed_seconds=time.perf_counter()-start,
        source_commit=subprocess.check_output(['D:/program/Git/cmd/git.exe', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()))
    print('All training complete; assessment not scored; seed', a.seed, flush=True)


if __name__ == '__main__':
    main()
