import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import unittest
import numpy as np


class AlternatingOptimizationChecks(unittest.TestCase):
    def setUp(self):
        try:
            from ssa_task_srd.learning import tensorflow,build_proxy
            from ssa_task_srd.collaborative import build_alternating_task
            self.tf=tensorflow(42)
        except ImportError:
            self.skipTest('Optional TensorFlow dependency is absent')
        self.proxy=build_proxy(self.tf,72,24)
        self.task=build_alternating_task(self.tf,self.proxy,72,
            {'selector_epochs':3,'n_train':4,'batch_size':2,'learning_rate':.001})
        rng=np.random.default_rng(42)
        self.c=self.tf.constant(rng.normal(size=(2,4,72)),dtype=self.tf.float32)
        self.y=self.tf.constant(rng.normal(size=(2,24)),dtype=self.tf.float32)

    def test_feature_encoder_is_shared_and_parameter_groups_are_disjoint(self):
        self.assertIs(self.task.feature_extractor.layers[-1].kernel,self.proxy.layers[0].kernel)
        head={id(v) for v in self.task.head.trainable_variables}
        proxy={id(v) for v in self.proxy.trainable_variables}
        features={id(v) for v in self.task.feature_extractor.trainable_variables}
        self.assertFalse(head & proxy)
        self.assertTrue(features <= proxy)
        self.assertEqual(self.task.head.count_params(),33)

    def test_selector_phase_updates_only_head_and_returns_preupdate_reconstruction(self):
        p0=[w.copy() for w in self.proxy.get_weights()]
        h0=[w.copy() for w in self.task.head.get_weights()]
        rebuilt,_=self.task.reconstruct(self.c)
        loss,cached,norm=self.task.selector_step(self.c,self.y)
        self.assertTrue(np.isfinite(float(loss)));self.assertGreater(float(norm),0)
        for before,after in zip(p0,self.proxy.get_weights()):np.testing.assert_array_equal(before,after)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(h0,self.task.head.get_weights())))
        np.testing.assert_allclose(cached.numpy(),rebuilt.numpy(),atol=1e-6,rtol=1e-6)
        self.assertEqual(int(self.task.head_optimizer.iterations.numpy()),1)
        self.assertEqual(int(self.task.proxy_optimizer.iterations.numpy()),0)

    def test_proxy_phase_updates_shared_features_but_not_head(self):
        _,cached,_=self.task.selector_step(self.c,self.y)
        h0=[w.copy() for w in self.task.head.get_weights()]
        q0=[w.copy() for w in self.task.feature_extractor.get_weights()]
        output0=[w.copy() for w in self.proxy.layers[-1].get_weights()]
        loss,norm=self.task.proxy_step(cached,self.y)
        self.assertTrue(np.isfinite(float(loss)));self.assertGreater(float(norm),0)
        for before,after in zip(h0,self.task.head.get_weights()):np.testing.assert_array_equal(before,after)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(q0,self.task.feature_extractor.get_weights())))
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(output0,self.proxy.layers[-1].get_weights())))
        self.assertEqual(int(self.task.proxy_optimizer.iterations.numpy()),1)

    def test_restoring_both_parts_restores_shared_selector_and_proxy_outputs(self):
        p0=[w.copy() for w in self.proxy.get_weights()];h0=[w.copy() for w in self.task.head.get_weights()]
        before=self.task(self.c).numpy()
        reconstructed,_=self.task.reconstruct(self.c)
        _,cached,_=self.task.selector_step(self.c,self.y);self.task.proxy_step(cached,self.y)
        self.proxy.set_weights(p0);self.task.head.set_weights(h0)
        np.testing.assert_array_equal(self.task(self.c).numpy(),before)
        after,_=self.task.reconstruct(self.c)
        np.testing.assert_array_equal(after.numpy(),reconstructed.numpy())

    def test_task_style_head_zero_bias_and_max_norm_are_applied(self):
        from ssa_task_srd.collaborative import build_alternating_task
        task=build_alternating_task(self.tf,self.proxy,72,
            {'selector_epochs':3,'n_train':4,'batch_size':2,'learning_rate':.001,
             'selector_bias':0.0,'selector_max_norm':.25})
        np.testing.assert_array_equal(task.head.get_weights()[1],np.zeros(1))
        task.selector_step(self.c,self.y)
        self.assertLessEqual(float(np.linalg.norm(task.head.get_weights()[0])),.250001)

    def test_full_component_count_does_not_add_trainable_parameters(self):
        x=self.tf.zeros((2,145,72))
        rebuilt,w=self.task.reconstruct(x)
        self.assertEqual(tuple(rebuilt.shape),(2,72))
        self.assertEqual(tuple(w.shape),(2,145,1))
        self.assertEqual(self.task.head.count_params(),33)


if __name__=='__main__':unittest.main()
