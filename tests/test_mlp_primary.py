import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import unittest
import numpy as np


class PrimaryMLPChecks(unittest.TestCase):
    def setUp(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest('Optional TensorFlow dependency is absent')

    def test_global_gate_is_one_parameter_and_has_isolated_update_phases(self):
        from ssa_task_srd.learning import tensorflow,build_proxy
        from ssa_task_srd.mlp_primary import build_global_gate
        from ssa_task_srd.collaborative import equal_weights
        tf=tensorflow(7);proxy=build_proxy(tf,24,8)
        cfg={'selector_epochs':2,'batch_size':2,'learning_rate':.001}
        gate=build_global_gate(tf,proxy,cfg,4)
        x=np.arange(48,dtype=np.float32).reshape(2,24)/48;y=np.ones((2,8),dtype=np.float32)
        rebuilt,scale=gate.reconstruct(x)
        self.assertEqual(float(scale),.5)
        np.testing.assert_array_equal(rebuilt.numpy(),x*.5)
        p0=[a.copy() for a in proxy.get_weights()];l0=float(gate.logit.numpy())
        _,cached,_=gate.selector_step(x,y)
        self.assertTrue(equal_weights(p0,proxy.get_weights()))
        self.assertNotEqual(l0,float(gate.logit.numpy()))
        np.testing.assert_array_equal(cached.numpy(),x*.5)
        l1=float(gate.logit.numpy());gate.proxy_step(cached,y)
        self.assertEqual(l1,float(gate.logit.numpy()))
        self.assertFalse(equal_weights(p0,proxy.get_weights()))

    def test_branches_share_initial_proxy_and_update_budget_and_restore_checkpoint(self):
        from ssa_task_srd.learning import tensorflow,build_proxy,weight_digest
        from ssa_task_srd.mlp_primary import train_branch,predict
        tf=tensorflow(8);base=build_proxy(tf,24,8)
        initial=[a.copy() for a in base.get_weights()]
        rng=np.random.default_rng(8);x=rng.normal(size=(6,24));y=rng.normal(size=(6,8))
        cfg={'seed':8,'history':24,'selector_epochs':2,'batch_size':2,'learning_rate':.001,
             'selector_bias':0.,'selector_max_norm':.25}
        audits=[]
        for branch in ['raw_continue','global_gate','task_full']:
            proxy=build_proxy(tf,24,8);proxy.set_weights(initial)
            inputs=np.stack([.6*x,.4*x],axis=1) if branch=='task_full' else x
            gate,a=train_branch(tf,proxy,branch,inputs[:4],y[:4],inputs[4:],y[4:],cfg)
            audits.append(a)
            self.assertEqual(a['initial_proxy_sha256'],weight_digest(initial))
            self.assertEqual(a['proxy_optimizer_steps'],4)
            selected=float(np.square(predict(proxy,gate,inputs[4:])-y[4:]).mean())
            self.assertAlmostEqual(selected,a['best_selection_mse_z'],places=10)
            self.assertGreaterEqual(a['best_epoch'],0);self.assertLessEqual(a['best_epoch'],2)
            self.assertTrue(a['coupled_checkpoint_restored'])
        self.assertEqual(len({a['batch_order_sha256'] for a in audits}),1)


if __name__=='__main__':unittest.main()
