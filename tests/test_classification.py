import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import unittest
import numpy as np
from ssa_task_srd.classification import split_indices, full_ssa, scores


class ClassificationChecks(unittest.TestCase):
    def test_stratified_roles_are_disjoint_and_deterministic(self):
        y = np.array([0, 1]*100)
        a = split_indices(y, 37); b = split_indices(y, 37)
        self.assertEqual(sum(map(len, a.values())), len(y))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
            self.assertEqual(set(y[a[k]]), {0, 1})
        self.assertFalse(set(a['fit']) & set(a['selection']))
        self.assertFalse(set(a['assessment']) & (set(a['fit']) | set(a['selection'])))

    def test_ssa_exact_reconstruction_and_energy_rule(self):
        x = np.random.default_rng(5).normal(size=40)
        c, rank = full_ssa(x, 8, .9)
        np.testing.assert_allclose(c.sum(axis=0), x, atol=5e-7)
        s = np.linalg.svd(np.lib.stride_tricks.sliding_window_view(x, 8).T, compute_uv=False)
        energy = np.cumsum(s*s)/np.sum(s*s)
        self.assertGreaterEqual(energy[rank-1], .9)
        if rank > 1:
            self.assertLess(energy[rank-2], .9)

    def test_classification_metrics_use_logits_and_correct_direction(self):
        logits = np.array([[5., -5.], [-5., 5.]])
        a = scores(logits, np.array([0, 1])); b = scores(logits, np.array([1, 0]))
        self.assertEqual(a['accuracy'], 1.); self.assertEqual(a['macro_f1'], 1.)
        self.assertLess(a['cross_entropy'], b['cross_entropy'])

    def test_alternating_classification_has_real_ce_feedback_and_matched_budget(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest('TensorFlow absent')
        from ssa_task_srd.learning import tensorflow, weight_digest
        from ssa_task_srd.classification import build_classifier, train_branch, predict
        tf = tensorflow(9); base = build_classifier(tf, 24); initial = base.get_weights()
        rng = np.random.default_rng(9); x = rng.normal(size=(8, 24)).astype(np.float32)
        y = np.array([0, 1]*4)
        cfg = {'length': 24, 'collaborative_epochs': 2, 'batch_size': 2, 'learning_rate': .001,
               'selector_bias': 0., 'selector_max_norm': .25, 'branches': ['raw_continue', 'global_gate', 'task_full']}
        audits = []
        for branch in cfg['branches']:
            proxy = build_classifier(tf, 24); proxy.set_weights(initial)
            inputs = np.stack([.7*x, .3*x], axis=1) if branch == 'task_full' else x
            gate, a = train_branch(tf, proxy, branch, inputs[:6], y[:6], inputs[6:], y[6:], cfg, 9)
            self.assertEqual(a['initial_proxy_sha256'], weight_digest(initial))
            self.assertEqual(a['classifier_optimizer_steps'], 6)
            self.assertAlmostEqual(scores(predict(proxy, gate, inputs[6:]), y[6:])['cross_entropy'], a['best_selection_cross_entropy'], places=7)
            self.assertTrue(a['coupled_checkpoint_restored']); audits.append(a)
        self.assertEqual(len({a['batch_order_sha256'] for a in audits}), 1)

    def test_selector_normalization_preserves_reconstruction_amplitudes_and_zeros(self):
        from ssa_task_srd.learning import tensorflow
        from ssa_task_srd.classification import build_classifier, build_gate
        tf = tensorflow(21); proxy = build_classifier(tf, 24)
        cfg = dict(length=24, collaborative_epochs=2, batch_size=2, learning_rate=.001,
                   selector_bias=0., selector_max_norm=.25,
                   selector_input_normalization='component_rms', selector_rms_floor=.001)
        gate = build_gate(tf, proxy, 'task_full', cfg, 6)
        rng = np.random.default_rng(21)
        c = rng.normal(size=(2, 3, 24)).astype(np.float32)
        _, w = gate.reconstruct(c)
        scaled = c*np.array([.1, 2., 10.], np.float32)[None, :, None]
        reconstructed, ws = gate.reconstruct(scaled)
        np.testing.assert_allclose(w, ws, atol=1e-7)
        np.testing.assert_allclose(reconstructed, np.sum(scaled*ws.numpy(), axis=1), atol=1e-6)
        z, wz = gate.reconstruct(np.zeros_like(c))
        self.assertTrue(np.isfinite(wz).all()); np.testing.assert_array_equal(z, 0.)
        gate.restore([np.zeros((32, 1), np.float32), np.zeros(1, np.float32)])
        rebuilt, _ = gate.reconstruct(scaled)
        np.testing.assert_allclose(rebuilt, .5*scaled.sum(axis=1), atol=1e-6)

    def test_selector_normalization_default_preserves_legacy_behavior(self):
        from ssa_task_srd.learning import tensorflow
        from ssa_task_srd.classification import build_classifier, build_gate
        tf = tensorflow(22); proxy = build_classifier(tf, 24)
        cfg = dict(length=24, collaborative_epochs=2, batch_size=2, learning_rate=.001,
                   selector_bias=0., selector_max_norm=.25)
        gate = build_gate(tf, proxy, 'task_full', cfg, 6)
        x = np.random.default_rng(22).normal(size=(2, 3, 24)).astype(np.float32)
        rebuilt, w = gate.reconstruct(x)
        f = proxy.layers[0](x.reshape(-1, 24), training=False)
        expected = gate.head(f, training=False).numpy().reshape(2, 3, 1)
        np.testing.assert_array_equal(w, expected)
        np.testing.assert_array_equal(rebuilt, np.sum(x*expected, axis=1))

    def test_longer_budget_exactly_preserves_short_run_prefix(self):
        from ssa_task_srd.learning import tensorflow, weight_digest
        from ssa_task_srd.classification import build_classifier, train_branch
        tf = tensorflow(17); base = build_classifier(tf, 24); initial = base.get_weights()
        rng = np.random.default_rng(17); x = rng.normal(size=(8, 24)).astype(np.float32)
        y = np.array([0, 1]*4)
        cfg = {'length': 24, 'collaborative_epochs': 2, 'batch_size': 2, 'learning_rate': .001,
               'selector_bias': 0., 'selector_max_norm': .25, 'branches': ['raw_continue', 'global_gate', 'task_full']}
        for branch in cfg['branches']:
            inputs = np.stack([.7*x, .3*x], axis=1) if branch == 'task_full' else x
            proxy = build_classifier(tf, 24); proxy.set_weights(initial)
            _, short = train_branch(tf, proxy, branch, inputs[:6], y[:6], inputs[6:], y[6:], cfg, 17)
            proxy = build_classifier(tf, 24); proxy.set_weights(initial); captures = []
            def capture(record, bp, bg, lp, lg):
                captures.append((record.copy(), weight_digest(bp), weight_digest(bg)))
            extended = dict(cfg, collaborative_epochs=4, schedule_epochs=2, snapshot_epochs=[2, 4])
            _, long = train_branch(tf, proxy, branch, inputs[:6], y[:6], inputs[6:], y[6:], extended, 17, capture)
            self.assertEqual(captures[0][1], short['final_proxy_sha256'])
            self.assertEqual(captures[0][2], short['final_head_sha256'])
            self.assertEqual(captures[0][0]['batch_order_prefix_sha256'], short['batch_order_sha256'])
            self.assertEqual(long['classifier_optimizer_steps'], 12)
            self.assertLessEqual(long['best_selection_cross_entropy'], short['best_selection_cross_entropy'])


if __name__ == '__main__':
    unittest.main()
