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


if __name__=='__main__':unittest.main()
