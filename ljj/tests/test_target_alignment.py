"""Projection constraints for train-selected independent read directions."""

import unittest

import torch

from experiments.probe_target_alignment import (dot, downhill_direction,
                                                project_independent_read_bias)


class TargetAlignmentTests(unittest.TestCase):
    def test_projection_removes_tied_and_softmax_null_directions(self):
        gradient = torch.arange(20, dtype=torch.float64).square().reshape(5, 4)
        projected = project_independent_read_bias(gradient)
        torch.testing.assert_close(projected.sum(dim=0), torch.zeros(4, dtype=torch.float64),
                                   atol=1e-10, rtol=0)
        torch.testing.assert_close(projected.sum(dim=1), torch.zeros(5, dtype=torch.float64),
                                   atol=1e-10, rtol=0)
        self.assertGreater(projected.norm().item(), 0)

    def test_direction_is_downhill_only_in_independent_subspace(self):
        gradient = project_independent_read_bias(torch.randn(5, 4))
        direction = downhill_direction({"1": gradient})
        self.assertLess(dot({"1": gradient}, direction), 0)
        self.assertAlmostEqual(direction["1"].abs().max().item(), 1)
        with self.assertRaisesRegex(ValueError, "No finite"):
            downhill_direction({"1": torch.zeros(5, 4)})
        with self.assertRaisesRegex(ValueError, "Expected"):
            project_independent_read_bias(torch.zeros(4, 4))


if __name__ == "__main__":
    unittest.main()
