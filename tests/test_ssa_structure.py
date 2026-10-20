import unittest
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from ssa_task_srd.ssa import decompose
from ssa_task_srd.ssa_structure import decomposition_details, weighted_correlation, spectral_features, candidate_pair


class SSAStructureChecks(unittest.TestCase):
    def test_decomposition_matches_existing_ssa_and_reconstructs(self):
        t = np.arange(144)
        x = .002*t + np.sin(2*np.pi*t/24) + .2*np.cos(2*np.pi*t/12)
        c, s, w = decomposition_details(x, 48)
        np.testing.assert_allclose(c[:6], decompose(x, 48, 6)[:6], atol=1e-12)
        np.testing.assert_allclose(c.sum(axis=0), x, atol=1e-12)
        np.testing.assert_allclose(np.sum(s*s), np.sum(w*x*x), rtol=1e-12)

    def test_wcor_equals_embedded_hankel_inner_products(self):
        t = np.arange(96)
        c = np.array([np.sin(t/4), np.cos(t/7), np.ones(96), np.zeros(96)])
        w = np.convolve(np.ones(24), np.ones(73))
        actual = weighted_correlation(c, w)
        embedded = np.array([sliding_window_view(a, 24).ravel() for a in c])
        expected = np.zeros((4, 4))
        for i in range(3):
            for j in range(3):
                expected[i, j] = np.dot(embedded[i], embedded[j])/(np.linalg.norm(embedded[i])*np.linalg.norm(embedded[j]))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_known_period_and_residual_daily_amplitude(self):
        t = np.arange(432)/144
        daily = 2*np.sin(2*np.pi*t)
        x = 4+daily+.5*np.cos(4*np.pi*t)
        stats = spectral_features(x)
        self.assertAlmostEqual(stats['peak_period_hours'], 24)
        self.assertAlmostEqual(stats['daily_harmonic_power_fraction'], 1)
        self.assertAlmostEqual(stats['daily_amplitude_z'], 2)
        self.assertAlmostEqual(spectral_features(x-daily)['daily_amplitude_z'], 0, places=12)
        self.assertTrue(np.isnan(spectral_features(np.ones(432))['peak_period_hours']))

    def test_pair_rule_needs_all_three_indicators(self):
        thresholds = {'min_absolute_wcor': .8, 'max_relative_singular_value_gap': .2,
                      'require_same_positive_peak_bin': True}
        self.assertTrue(candidate_pair(10, 9, -.9, 3, 3, thresholds))
        self.assertFalse(candidate_pair(10, 6, .9, 3, 3, thresholds))
        self.assertFalse(candidate_pair(10, 9, .7, 3, 3, thresholds))
        self.assertFalse(candidate_pair(10, 9, .9, 3, 6, thresholds))


if __name__ == '__main__':
    unittest.main()
