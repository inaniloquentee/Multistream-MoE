"""Small causal-probe checks independent of the H20 checkpoint."""

import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.qwen_diagnostics import (activation_probe, diagnose, replace_maps,
                                          set_initial_expert_read_contrast,
                                          set_initial_stream_contrast)
from experiments.train_qwen import forward_batch
from multistream_moe.qwen_reference import QwenMultiStream


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8,
                          max_position_embeddings=32, tie_word_embeddings=True)
        cfg._attn_implementation = "sdpa"
        self.tokens = torch.randint(0, 128, (2, 8))
        self.model = QwenMultiStream(Qwen3ForCausalLM(cfg).eval(), "D").eval()
        self.validation = [(self.tokens, torch.roll(self.tokens, -1, dims=1))] * 2

    def test_initial_read_maps_have_no_effect(self):
        result = diagnose(self.model, "D", self.validation, forward_batch, "fp32")
        self.assertAlmostEqual(result["read_tied"]["delta_nll"], 0, places=5)
        self.assertAlmostEqual(result["read_uniform"]["delta_nll"], 0, places=5)
        self.assertEqual(result["activation"]["1"]["input_centered_rms"], 0)

    def test_read_intervention_restores_all_parameters_on_failure(self):
        with torch.no_grad():
            self.model.moes["3"].read_weight[1, 0, 0] = 1.0
        original = self.model.moes["3"].read_weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with replace_maps(self.model, "read", "uniform"):
                self.assertEqual(self.model.moes["3"].read_weight.abs().sum().item(), 0)
                raise RuntimeError("probe")
        torch.testing.assert_close(self.model.moes["3"].read_weight, original)
        measured = activation_probe(self.model, self.tokens, "fp32")
        self.assertGreater(measured["3"]["read_weight_expert_delta_rms"], 0)
        self.assertGreater(measured["3"]["read_signal_fp32_same_input_rms"], 0)
        self.assertAlmostEqual(measured["3"]["read_signal_fp32_same_input_rms"],
                               measured["3"]["read_expert_delta_rms"], places=5)

    def test_contrast_preserves_lift_and_increases_read_input_signal(self):
        with torch.no_grad():
            before = self.model(self.tokens)[0]
            original_signal = activation_probe(self.model, self.tokens, "fp32")["3"]["input_centered_rms"]
            set_initial_stream_contrast(self.model, 0.2)
            after = self.model(self.tokens)[0]
            new_signal = activation_probe(self.model, self.tokens, "fp32")["3"]["input_centered_rms"]
        torch.testing.assert_close(after, before, atol=2e-4, rtol=2e-4)
        self.assertGreater(new_signal, 15 * original_signal)

    def test_expert_read_seed_opens_second_layer_only(self):
        set_initial_stream_contrast(self.model, 0.2)
        baseline = activation_probe(self.model, self.tokens, "fp32")
        self.assertEqual(baseline["3"]["read_expert_delta_rms"], 0)
        set_initial_expert_read_contrast(self.model, 0.25)
        seeded = activation_probe(self.model, self.tokens, "fp32")
        self.assertLess(seeded["1"]["read_expert_delta_rms"], 1e-7)
        self.assertGreater(seeded["3"]["read_signal_fp32_same_input_rms"], 0)
        self.assertAlmostEqual(seeded["3"]["read_signal_fp32_same_input_rms"],
                               seeded["3"]["read_expert_delta_rms"], places=5)
        with self.assertRaisesRegex(ValueError, "before training"):
            set_initial_expert_read_contrast(self.model, 0.25)

    def test_failed_seed_does_not_partially_update_other_layers(self):
        with torch.no_grad():
            self.model.moes["3"].read_weight[1, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "before training"):
            set_initial_expert_read_contrast(self.model, 0.25)
        self.assertEqual(self.model.moes["1"].read_bias.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
