"""Check decomposition of D gradients into nested C and extra expert directions."""

import unittest

import torch

from experiments.probe_read_aux_gradients import component, cosine, dot


class ReadGradientComponentTests(unittest.TestCase):
    def test_shared_and_independent_are_orthogonal(self):
        gradient = {"a": torch.arange(20, dtype=torch.float32).reshape(5, 4)}
        shared = component(gradient, False)
        independent = component(gradient, True)
        torch.testing.assert_close(shared["a"] + independent["a"], gradient["a"])
        self.assertAlmostEqual(dot(shared, independent), 0)
        self.assertAlmostEqual(cosine(independent, independent), 1)


if __name__ == "__main__":
    unittest.main()
