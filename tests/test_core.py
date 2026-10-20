import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import unittest
import numpy as np
from ssa_task_srd.ssa import decompose, ordinary
from ssa_task_srd.repair import repair
from ssa_task_srd.data import fit_scale, inject, make_windows


class ScientificChecks(unittest.TestCase):
    def test_ssa_identity_with_residual_group(self):
        x=np.random.default_rng(1).normal(size=80)
        c=decompose(x,20,4)
        np.testing.assert_allclose(c.sum(axis=0),x,atol=1e-12)

    def test_ssa_rank_two_periodic_reconstruction(self):
        x=np.sin(2*np.pi*np.arange(96)/24)
        recovered=ordinary(decompose(x,24,2))
        np.testing.assert_allclose(recovered,x,atol=1e-10)

    def test_scaler_uses_only_training_observations(self):
        x=np.arange(100,dtype=float)
        a=fit_scale(x,70)
        x[70:]=1e9
        self.assertEqual(a,fit_scale(x,70))

    def test_target_ranges_do_not_cross_split_boundaries(self):
        x=np.arange(1200,dtype=float)
        config={'history':72,'horizon':24,'stride':24}
        windows,b=make_windows(x,x,np.zeros(len(x),dtype=bool),config)
        for o in windows['train']['origins']:self.assertLessEqual(o+24,b['train_end'])
        for o in windows['val']['origins']:
            self.assertGreaterEqual(o,b['train_end']);self.assertLessEqual(o+24,b['val_end'])
        for o in windows['test']['origins']:self.assertGreaterEqual(o,b['val_end'])

    def test_future_changes_cannot_change_current_ssa_input(self):
        x=np.sin(np.arange(1200)/10)
        mask=np.zeros(len(x),dtype=bool)
        config={'history':72,'horizon':24,'stride':24}
        first,_=make_windows(x,x,mask,config)
        o=first['val']['origins'][0]
        changed=x.copy();changed[o:]+=1000
        second,_=make_windows(changed,changed,mask,config)
        np.testing.assert_array_equal(first['val']['x'][0],second['val']['x'][0])
        np.testing.assert_allclose(decompose(first['val']['x'][0],24,4),decompose(second['val']['x'][0],24,4))

    def test_repair_preserves_constant_and_does_not_mutate_input(self):
        x=np.ones(96)*7
        np.testing.assert_array_equal(repair(x,24),x)
        spike=x.copy();spike[48]=100
        before=spike.copy()
        repaired=repair(spike,24,k=3)
        np.testing.assert_array_equal(spike,before)
        self.assertLess(abs(repaired[48]-7),abs(spike[48]-7))

    def test_perturbation_reproducible_and_reference_untouched(self):
        x=np.zeros(100)
        config={'seed':42,'noise_std':.1,'outlier_rate':.03,'outlier_scale':3,'outlier_length':3}
        a,ma=inject(x,config);b,mb=inject(x,config)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ma,mb)
        np.testing.assert_array_equal(x,np.zeros(100))

    def test_bad_ssa_shapes_are_rejected(self):
        with self.assertRaises(ValueError):decompose(np.ones(20),30,2)
        with self.assertRaises(ValueError):decompose(np.ones((2,20)),5,2)

    def test_validation_only_does_not_construct_test_windows(self):
        x=np.arange(1200,dtype=float)
        config={'history':72,'horizon':24,'stride':24,'evaluation_splits':['val']}
        windows,b=make_windows(x,x,np.zeros(len(x),dtype=bool),config)
        self.assertEqual(set(windows),{'train','val'})
        changed=x.copy();changed[b['val_end']:]=1e12
        other,_=make_windows(changed,changed,np.zeros(len(x),dtype=bool),config)
        for split in windows:
            for field in ['x','reference','y','origins']:
                np.testing.assert_array_equal(windows[split][field],other[split][field])

    def test_corruption_seed_is_independent_of_training_seed(self):
        config={'seed':42,'corruption_seed':7,'noise_std':.1,'outlier_rate':.03,'outlier_scale':3,'outlier_length':3}
        a,ma=inject(np.zeros(100),config)
        config['seed']=44
        b,mb=inject(np.zeros(100),config)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ma,mb)

    def test_feature_controls_have_matched_trainable_heads_and_frozen_features(self):
        try:
            from ssa_task_srd.learning import tensorflow,build_proxy,build_feature_selector,weight_digest
            tf=tensorflow(42)
        except ImportError:
            self.skipTest('Optional TensorFlow dependency is absent')
        proxy=build_proxy(tf,72,24)
        initial=[w.copy() for w in proxy.layers[0].get_weights()]
        # Simulate a learned feature extractor, keeping the saved random control distinct.
        proxy.layers[0].set_weights([w+.01 for w in initial]);proxy.trainable=False
        frozen=[w.copy() for w in proxy.get_weights()]
        selectors=[build_feature_selector(tf,proxy,72,mode,initial)
            for mode in ['proxy_features','random_features']]
        self.assertEqual([sum(np.prod(v.shape) for v in s.trainable_variables) for s in selectors],[33,33])
        self.assertEqual(*[weight_digest([v.numpy() for v in s.trainable_variables]) for s in selectors])
        c=tf.constant(np.random.default_rng(1).normal(size=(2,4,72)),dtype=tf.float32)
        for selector in selectors:
            frozen_selector=[v.numpy().copy() for v in selector.non_trainable_variables]
            with tf.GradientTape() as tape:
                w=tf.reshape(selector(tf.reshape(c,[-1,72,1])),[2,4,1])
                loss=tf.reduce_mean(proxy(tf.reduce_sum(c*w,axis=1))**2)
            gradients=tape.gradient(loss,selector.trainable_variables)
            self.assertTrue(all(g is not None for g in gradients))
            self.assertGreater(sum(float(tf.reduce_sum(tf.abs(g))) for g in gradients),0)
            tf.keras.optimizers.Adam(.001).apply_gradients(zip(gradients,selector.trainable_variables))
            self.assertTrue(all(np.array_equal(a,b.numpy()) for a,b in zip(frozen_selector,selector.non_trainable_variables)))
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(frozen,proxy.get_weights())))

    def test_forecasting_feedback_reaches_selector_not_proxy(self):
        try:
            from ssa_task_srd.learning import tensorflow,build_proxy,build_selector
            tf=tensorflow(42)
        except ImportError:
            self.skipTest('Optional TensorFlow dependency is absent')
        proxy=build_proxy(tf,72,24);proxy.trainable=False
        selector=build_selector(tf,72)
        c=tf.constant(np.random.default_rng(42).normal(size=(2,4,72)),dtype=tf.float32)
        frozen=[w.copy() for w in proxy.get_weights()]
        with tf.GradientTape() as tape:
            weights=tf.reshape(selector(tf.reshape(c,[-1,72,1])),[2,4,1])
            predicted=proxy(tf.reduce_sum(c*weights,axis=1))
            loss=tf.reduce_mean(predicted**2)
        gradients=tape.gradient(loss,selector.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertGreater(sum(float(tf.reduce_sum(tf.abs(g))) for g in gradients),0)
        tf.keras.optimizers.Adam(.001).apply_gradients(zip(gradients,selector.trainable_variables))
        self.assertFalse(proxy.trainable_variables)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(frozen,proxy.get_weights())))


if __name__=='__main__':unittest.main()
