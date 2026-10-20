"""Prespecified FordA TRAIN-only pilot; never downloads or reads official TEST."""
import os
for key in ['TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.classification import split_indices, full_ssa, build_classifier, build_gate, scores, pretrain, train_branch, predict
from ssa_task_srd.learning import tensorflow, weight_digest, weight_summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/classification_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--protocol', default='configs/classification_protocol.json')
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    out = root/args.out; out.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    cfg = json.loads((root/args.protocol).read_text(encoding='utf-8'))
    pretrained_run = root/cfg['pretrained_from'] if cfg.get('pretrained_from') else None
    original_hashes = {}
    if pretrained_run is not None:
        original_hashes = {str(p.relative_to(pretrained_run)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in pretrained_run.rglob('*') if p.is_file()}
        source_cfg = json.loads((pretrained_run/'protocol.json').read_text())
        for key, value in source_cfg.items():
            if key not in ['collaborative_epochs', 'budget', 'stop_rule', 'scope']:
                assert cfg[key] == value, f'Unexpected protocol change: {key}'
        assert cfg['schedule_epochs'] == source_cfg['collaborative_epochs']
    write_json(out/'protocol.json', cfg)
    source = root/'data/forda/FordA_TRAIN.tsv'
    if not source.exists():
        raise FileNotFoundError('Run scripts/fetch_forda.py first')
    blob = source.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == cfg['train_sha256'], 'Dataset version differs from locked protocol'
    data = np.loadtxt(source, delimiter='\t')
    assert data.shape == (3601, 501) and np.isfinite(data).all()
    assert set(data[:, 0]) == {-1., 1.}
    labels = (data[:, 0] == 1).astype(np.int64); raw = data[:, 1:]
    split = split_indices(labels, cfg['split_seed'])
    scale_mean = float(raw[split['fit']].mean()); scale_std = float(raw[split['fit']].std())
    assert scale_std > 0
    x = ((raw-scale_mean)/scale_std).astype(np.float32)
    row_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in raw]
    for a, b in [('fit', 'selection'), ('fit', 'assessment'), ('selection', 'assessment')]:
        assert not set(split[a]) & set(split[b])
        assert not {row_hashes[i] for i in split[a]} & {row_hashes[i] for i in split[b]}, 'Duplicate waveforms across splits'
    # SSA is label-free and sample-local. Cache includes exact data/config/source identity.
    cache_hash = hashlib.sha256(blob+(root/'ssa_task_srd/ssa.py').read_bytes()+
        (root/'ssa_task_srd/classification.py').read_bytes()+
        json.dumps({'embedding': cfg['embedding'], 'energy': cfg['energy_threshold'], 'mean': scale_mean, 'std': scale_std}, sort_keys=True).encode()).hexdigest()
    cache = root/f'data/forda/ssa_{cache_hash[:20]}.npz'
    if cache.exists():
        with np.load(cache) as z:
            components = z['components']; ranks = z['ranks']; fixed = z['fixed']
    else:
        components = np.empty((len(x), cfg['embedding']+1, cfg['length']), dtype=np.float32)
        ranks = np.empty(len(x), dtype=np.int64); fixed = np.empty_like(x)
        for i, row in enumerate(x):
            c, rank = full_ssa(row.astype(np.float64), cfg['embedding'], cfg['energy_threshold'])
            components[i] = c; ranks[i] = rank; fixed[i] = c[:rank].sum(axis=0)
            if (i+1) % 500 == 0:
                print(f'SSA {i+1}/{len(x)}', flush=True)
        np.savez_compressed(cache, components=components, ranks=ranks, fixed=fixed)
    np.testing.assert_allclose(components.sum(axis=1), x, atol=2e-6, rtol=2e-6)
    inputs = {'raw_continue': x, 'half_continue': x*.5, 'ssa_energy_continue': fixed,
              'global_gate': x, 'task_full': components}
    audit = {'dataset': 'FordA official TRAIN only', 'source_url': cfg['train_url'],
        'sha256': hashlib.sha256(blob).hexdigest(), 'shape': list(data.shape),
        'indices': {k: v.tolist() for k, v in split.items()}, 'counts': {k: len(v) for k, v in split.items()},
        'class_counts': {k: np.bincount(labels[ix], minlength=2).tolist() for k, ix in split.items()},
        'label_mapping': {'-1': 0, '1': 1}, 'scale_mean': scale_mean, 'scale_std': scale_std,
        'scale_fit_role': 'fit only', 'no_duplicate_waveforms_across_splits': True,
        'official_test_downloaded': False, 'official_test_read': False,
        'no_subject_identifiers_available': True, 'artificial_pollution': False, 'SRD': False,
        'embedding': cfg['embedding'], 'component_count_with_residual': cfg['embedding']+1,
        'cache': str(cache.relative_to(root)), 'cache_signature': cache_hash,
        'fixed_ssa_rank_by_split': {k: {'min': int(ranks[v].min()), 'median': float(np.median(ranks[v])),
                                       'max': int(ranks[v].max())} for k, v in split.items()}}
    audit['assessment_previously_explored'] = bool(cfg.get('assessment_previously_explored', False))
    write_json(out/'data_audit.json', audit)
    if pretrained_run is not None:
        prior_audit = json.loads((pretrained_run/'data_audit.json').read_text())
        for key in ['sha256', 'indices', 'counts', 'class_counts', 'scale_mean', 'scale_std', 'embedding']:
            assert audit[key] == prior_audit[key], f'Data role changed: {key}'
    np.savez_compressed(out/'samples.npz', **{k: v for k, v in split.items()}, labels=labels, fixed_ranks=ranks)
    print('Data roles:', audit['counts'], 'class counts:', audit['class_counts'], flush=True)
    manifests = []; audits = []
    for seed in cfg['seeds']:
        tf = tensorflow(seed); tf.keras.backend.clear_session()
        base = build_classifier(tf, cfg['length']); tick = time.perf_counter()
        print(f'Preparing common pretrained model seed={seed}', flush=True)
        if pretrained_run is None:
            training = pretrain(tf, base, x[split['fit']], labels[split['fit']], x[split['selection']], labels[split['selection']], cfg, seed)
        else:
            prior = pretrained_run/f'raw_pretrained_seed{seed}'
            base.load_weights(str(prior/'proxy.weights.h5'))
            training = json.loads((prior/'training.json').read_text())
            assert weight_digest(base.get_weights()) == training['initial_for_branches_sha256']
            np.testing.assert_allclose(scores(predict(base, None, x[split['selection']]), labels[split['selection']])['cross_entropy'],
                training['best_selection_cross_entropy'], atol=1e-7)
            training.update(reused_pretraining=True, reused_from=str(prior.relative_to(root)),
                reused_checkpoint_sha256=hashlib.sha256((prior/'proxy.weights.h5').read_bytes()).hexdigest())
        initial = base.get_weights(); initial_hash = weight_digest(initial)
        folder = out/f'raw_pretrained_seed{seed}'; folder.mkdir()
        base.save_weights(str(folder/'proxy.weights.h5'))
        training.update(seed=seed, initial_for_branches_sha256=initial_hash, elapsed_seconds=time.perf_counter()-tick)
        write_json(folder/'training.json', training)
        manifests.append({'branch': 'raw_pretrained', 'seed': seed, 'folder': folder.name})
        del base
        for branch in cfg['branches']:
            tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length']); proxy.set_weights(initial)
            tick = time.perf_counter(); print(f'Training seed={seed} branch={branch}', flush=True)
            a = inputs[branch]
            folder = out/f'{branch}_seed{seed}'; folder.mkdir()
            def snapshot(record, best_proxy, best_gate, last_proxy, last_gate):
                payload = {f'proxy_w{i}': v for i, v in enumerate(best_proxy)}
                payload.update({f'gate_w{i}': v for i, v in enumerate(best_gate)})
                np.savez_compressed(folder/f'epoch{record["epoch"]}_best.npz', **payload)
                if record['epoch'] == cfg['collaborative_epochs']:
                    payload = {f'proxy_w{i}': v for i, v in enumerate(last_proxy)}
                    payload.update({f'gate_w{i}': v for i, v in enumerate(last_gate)})
                    np.savez_compressed(folder/f'epoch{record["epoch"]}_last.npz', **payload)
                if pretrained_run is not None and record['epoch'] == source_cfg['collaborative_epochs']:
                    old = json.loads((pretrained_run/f'{branch}_seed{seed}'/'training.json').read_text())
                    for new_key, old_key in [('best_proxy_sha256', 'final_proxy_sha256'), ('best_head_sha256', 'final_head_sha256'),
                        ('batch_order_prefix_sha256', 'batch_order_sha256')]:
                        assert record[new_key] == old[old_key], f'First ten epochs not reproduced: {seed} {branch} {new_key}'
                    np.testing.assert_allclose(record['best_selection_cross_entropy'], old['best_selection_cross_entropy'], atol=1e-10)
                    record['old_ten_epoch_best_exactly_reproduced'] = True
            gate, training = train_branch(tf, proxy, branch, a[split['fit']], labels[split['fit']],
                a[split['selection']], labels[split['selection']], cfg, seed, snapshot_callback=snapshot)
            assert training['initial_proxy_sha256'] == initial_hash
            proxy.save_weights(str(folder/'proxy.weights.h5'))
            if gate:
                np.savez_compressed(folder/'gate.npz', **{f'w{i}': v for i, v in enumerate(gate.weights())})
            training['elapsed_seconds'] = time.perf_counter()-tick
            write_json(folder/'training.json', training); audits.append(training)
            manifests.append({'branch': branch, 'seed': seed, 'folder': folder.name})
            # Selection predictions are verified again when models are reloaded.
            np.savez_compressed(folder/'selection_logits.npz', logits=predict(proxy, gate, a[split['selection']]))
            write_json(out/'manifest.json', manifests)
            del proxy, gate
    for seed in cfg['seeds']:
        matched = [a for a in audits if a['seed'] == seed]
        for key in ['initial_proxy_sha256', 'batch_order_sha256', 'classifier_optimizer_steps', 'classifier_parameters']:
            assert len({a[key] for a in matched}) == 1, key
    # First assessment-label scoring occurs only after all prespecified training is complete.
    rows = []; checks = []; weights_summary = {}
    for item in manifests:
        seed = item['seed']; branch = item['branch']; folder = out/item['folder']
        tf.keras.backend.clear_session(); proxy = build_classifier(tf, cfg['length'])
        proxy.load_weights(str(folder/'proxy.weights.h5'))
        gate = build_gate(tf, proxy, branch, cfg, len(split['fit'])) if branch in ['global_gate', 'task_full'] else None
        if gate:
            with np.load(folder/'gate.npz') as z:
                gate.restore([z[f'w{i}'] for i in range(len(z.files))])
        branch_x = x if branch == 'raw_pretrained' else inputs[branch]
        predictions = {}
        for role, ix in split.items():
            logits = predict(proxy, gate, branch_x[ix], cfg['batch_size']); predictions[role] = logits
            score = scores(logits, labels[ix]); cm = score.pop('confusion_matrix')
            rows.append(dict(branch=branch, seed=seed, role=role, **score, confusion_matrix=json.dumps(cm)))
            if role == 'selection' and branch != 'raw_pretrained':
                saved = np.load(folder/'selection_logits.npz')['logits']
                np.testing.assert_allclose(logits, saved, atol=2e-6, rtol=2e-6)
                training = json.loads((folder/'training.json').read_text())
                np.testing.assert_allclose(score['cross_entropy'], training['best_selection_cross_entropy'], atol=1e-7)
                assert weight_digest(proxy.get_weights()) == training['final_proxy_sha256']
                assert weight_digest(gate.weights() if gate else []) == training['final_head_sha256']
        np.savez_compressed(folder/'logits.npz', **predictions)
        checks.append(dict(branch=branch, seed=seed, checkpoint_reloaded=True, selection_match=True))
        if branch == 'task_full':
            all_weights = {}
            for role, ix in split.items():
                chunks = [gate.reconstruct(branch_x[ix[s:s+cfg['batch_size']]])[1].numpy()[..., 0]
                          for s in range(0, len(ix), cfg['batch_size'])]
                all_weights[role] = np.concatenate(chunks)
            np.savez_compressed(folder/'component_weights.npz', **all_weights)
            weights_summary[str(seed)] = {k: weight_summary(v) for k, v in all_weights.items()}
        print(f'Reloaded {branch} seed={seed}: assessment accuracy={rows[-1]["accuracy"]:.4f}', flush=True)
    majority = int(np.bincount(labels[split['fit']]).argmax())
    for role, ix in split.items():
        logits = np.zeros((len(ix), 2)); logits[:, majority] = 1
        sc = scores(logits, labels[ix]); cm = sc.pop('confusion_matrix')
        rows.append(dict(branch='majority_fit_class', seed=None, role=role, **sc, confusion_matrix=json.dumps(cm)))
    frame = pd.DataFrame(rows); frame.to_csv(out/'metrics.csv', index=False)
    neural = frame[frame.branch != 'majority_fit_class']
    summary = neural.groupby(['role', 'branch'])[['accuracy', 'balanced_accuracy', 'macro_f1', 'cross_entropy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]; summary.to_csv(out/'summary.csv')
    paired = []
    for role in split:
        wide = neural[neural.role == role].pivot(index='seed', columns='branch', values='accuracy')
        for control in ['raw_continue', 'half_continue', 'ssa_energy_continue', 'global_gate']:
            for seed, delta in (wide.task_full-wide[control]).items():
                paired.append(dict(role=role, control=control, seed=seed, task_minus_control_accuracy_pp=float(delta*100)))
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'reload_checks.csv', index=False)
    write_json(out/'training_audit.json', {'runs': audits, 'all_matched_controls_passed': True,
        'all_saved_models_reloaded': True, 'assessment_first_scored_after_all_training': True,
        'test_read': False, 'retention_summary': weights_summary})
    write_json(out/'metadata.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'elapsed_seconds': time.perf_counter()-start, 'tensorflow': tf.__version__, 'numpy': np.__version__,
        'completed_models': len(manifests), 'scope': cfg['scope']})
    if pretrained_run is not None:
        assert all(hashlib.sha256((pretrained_run/p).read_bytes()).hexdigest() == digest for p, digest in original_hashes.items())
        write_json(out/'source_run_unchanged.json', {'source': str(pretrained_run.relative_to(root)),
            'files_checked': len(original_hashes), 'all_original_files_unchanged': True,
            'pretraining_reused': True, 'first_ten_epoch_prefix_exactly_checked_for_all_branches': True})
    print(summary.loc['assessment'].to_string(), flush=True)
    print('Classification pilot finished; official TEST remains unopened.', flush=True)


if __name__ == '__main__':
    main()
