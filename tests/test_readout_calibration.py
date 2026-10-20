import unittest
import numpy as np
from ssa_task_srd.readout_calibration import expanding_folds, pipeline, select_alpha


class ReadoutCalibrationChecks(unittest.TestCase):
    def test_expanding_folds_are_disjoint_ordered_and_use_past_labels(self):
        folds = expanding_folds(251)
        self.assertEqual([(len(t), len(v)) for t, v in folds], [(100, 50), (150, 50), (200, 51)])
        origins = 432+144*np.arange(251)
        for train, val in folds:
            self.assertLess(max(train), min(val))
            self.assertLessEqual(max(origins[train]+144), min(origins[val]))
        np.testing.assert_array_equal(np.concatenate([v for _, v in folds]), np.arange(100, 251))

    def test_scaler_is_fit_on_inner_training_prefix_only(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(20, 12)); y = rng.normal(size=(20, 4))
        x[12:] += 1000
        model = pipeline(10).fit(x[:12], y[:12])
        np.testing.assert_allclose(model['scale'].mean_, x[:12].mean(axis=0), atol=1e-12)
        self.assertGreater(abs(model['scale'].transform(x[12:]).mean()), 100)

    def test_constant_scaling_is_removed_before_selection_and_prediction(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(40, 18)); y = rng.normal(size=(40, 6))
        folds = expanding_folds(30)
        a, _, scores = select_alpha(x[:30], y[:30], folds, [.1, 1, 10, 100])
        b, _, scaled_scores = select_alpha(x[:30]*.51, y[:30], folds, [.1, 1, 10, 100])
        self.assertEqual(a, b)
        np.testing.assert_allclose(list(scores.values()), list(scaled_scores.values()), atol=1e-10, rtol=0)
        raw = pipeline(a).fit(x[:30], y[:30]).predict(x[30:])
        scaled = pipeline(b).fit(x[:30]*.51, y[:30]).predict(x[30:]*.51)
        np.testing.assert_allclose(raw, scaled, atol=1e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
