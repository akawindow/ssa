import unittest
from unittest.mock import patch
import numpy as np
from ssa_task_srd.data import observed_input
from ssa_task_srd.ssa import decompose
from ssa_task_srd.ssa_diagnostics import regroup,fixed_variants


class OriginalObservationSSAChecks(unittest.TestCase):
    def test_reference_condition_never_calls_injection(self):
        x=np.arange(96,dtype=float)
        with patch('ssa_task_srd.data.inject',side_effect=AssertionError('Must not inject')):
            observed,mask=observed_input(x,{'input_condition':'reference','noise_std':0.,'outlier_rate':0.})
        np.testing.assert_array_equal(observed,x)
        self.assertFalse(mask.any());self.assertIsNot(observed,x)

    def test_reference_condition_rejects_accidental_corruption_settings(self):
        for key in ['noise_std','outlier_rate']:
            config={'input_condition':'reference','noise_std':0.,'outlier_rate':0.};config[key]=.1
            with self.assertRaises(ValueError):observed_input(np.ones(96),config)

    def test_full_rank_and_residual_deletion_positive_controls(self):
        x=np.random.default_rng(1).normal(size=(2,96))
        full=np.array([decompose(a,24,24) for a in x])
        variants=fixed_variants(x,full,[2,6,24],6)
        np.testing.assert_allclose(variants['all_components_identity'],x,atol=1e-12)
        np.testing.assert_allclose(variants['ssa_rank_24'],x,atol=1e-12)
        np.testing.assert_array_equal(variants['drop_residual_after_6'],variants['ssa_rank_6'])
        old=np.array([decompose(a,24,6) for a in x])
        np.testing.assert_allclose(regroup(x,full,6),old,atol=1e-12)

    def test_single_component_deletion_preserves_every_other_group(self):
        x=np.random.default_rng(2).normal(size=(2,96))
        full=np.array([decompose(a,24,24) for a in x])
        grouped=regroup(x,full,6);before=full.copy()
        variants=fixed_variants(x,full,[6],6)
        for i in range(6):
            remaining=np.delete(grouped,i,axis=1).sum(axis=1)
            np.testing.assert_allclose(variants[f'drop_C{i+1}'],remaining,atol=1e-12)
        np.testing.assert_array_equal(full,before)

    def test_direct_ridge_readout_preserves_full_rank_identity(self):
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(3)
        x=rng.normal(size=(12,72));y=rng.normal(size=(12,24))
        components=np.array([decompose(a,24,24) for a in x])
        full=components[:,:-1,:].sum(axis=1)
        raw_model=Ridge(alpha=1,solver='cholesky').fit(x[:8],y[:8])
        full_model=Ridge(alpha=1,solver='cholesky').fit(full[:8],y[:8])
        np.testing.assert_allclose(raw_model.predict(x[8:]),full_model.predict(full[8:]),atol=1e-10,rtol=0)

    def test_constant_scaling_matches_ridge_when_penalty_is_scaled(self):
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(4)
        x=rng.normal(size=(20,24));y=rng.normal(size=(20,8));scale=.51
        raw=Ridge(alpha=1,solver='cholesky').fit(x[:16],y[:16])
        matched=Ridge(alpha=scale**2,solver='cholesky').fit(x[:16]*scale,y[:16])
        np.testing.assert_allclose(raw.predict(x[16:]),matched.predict(x[16:]*scale),atol=1e-10,rtol=0)


if __name__=='__main__':unittest.main()
