"""The saved-checkpoint metric distinguishes a clone from specialized experts."""

import unittest

import torch

from experiments.check_specialist_diversity import diversity


class SpecialistDiversityTests(unittest.TestCase):
    def test_identical_and_divergent_experts(self):
        parameters = {f"moes.3.experts.{j}.mlp.down_proj.weight": torch.ones(2, 2)
                      for j in range(1, 5)}
        self.assertEqual(diversity(parameters, "3")["routed_expert_centered_rms"], 0)
        parameters["moes.3.experts.4.mlp.down_proj.weight"] += 1
        self.assertGreater(diversity(parameters, "3")["routed_expert_centered_rms"], 0)


if __name__ == "__main__":
    unittest.main()
