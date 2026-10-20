import json
from pathlib import Path
import unittest
import numpy as np
from ssa_task_srd.java_srd import JavaSRD


class OriginalJavaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        if not (root/'data/java/runtime.json').is_file():
            raise unittest.SkipTest('Java runtime not configured; run scripts/setup_java.py first')
        cls.java=JavaSRD(root)

    def test_input_not_mutated_and_constant_preserved(self):
        x=np.ones((2,96))*7;before=x.copy()
        repaired,_=self.java.repair_batch(x,24,6.0,5)
        np.testing.assert_array_equal(x,before)
        np.testing.assert_array_equal(repaired,x)

    def test_batch_windows_do_not_influence_each_other(self):
        rng=np.random.default_rng(1)
        x=np.sin(2*np.pi*np.arange(96)/24)+rng.normal(0,.01,96);x[45]+=8
        alone,_=self.java.repair_batch(x[None,:],24,3.0,5)
        batch,_=self.java.repair_batch(np.stack([x,rng.normal(100,50,96)]),24,3.0,5)
        np.testing.assert_array_equal(alone[0],batch[0])
        self.assertLess(abs(alone[0,45]-np.sin(2*np.pi*45/24)),abs(x[45]-np.sin(2*np.pi*45/24)))

    def test_invalid_inputs_rejected_without_java_launch(self):
        for x,period in [(np.ones((2,12)),24),(np.ones((2,96)),23),(np.full((2,96),np.nan),24)]:
            with self.assertRaises(ValueError):self.java.repair_batch(x,period)


if __name__=='__main__':unittest.main()
