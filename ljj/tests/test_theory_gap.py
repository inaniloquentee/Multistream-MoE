"""Small Qwen lift must retain only one stream contrast direction initially."""

import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.check_theory_gap import probe
from experiments.qwen_diagnostics import set_initial_stream_contrast
from multistream_moe.qwen_reference import QwenMultiStream


class TheoryGapTests(unittest.TestCase):
    def test_reachable_states_and_counterexample_read(self):
        torch.manual_seed(9)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8,
                          max_position_embeddings=32, tie_word_embeddings=True)
        cfg._attn_implementation = "sdpa"
        model = QwenMultiStream(Qwen3ForCausalLM(cfg).eval(), "D").eval()
        tokens = torch.randint(0, 128, (2, 8))
        for contrast in (0.01, 0.2):
            set_initial_stream_contrast(model, contrast)
            result = probe(model, tokens)
            self.assertEqual(result["second_moe_expert_read_max_abs_difference_at_init"], 0)
            self.assertLess(result["layers"]["3"]["entry"]["off_axis_rms"], 1e-6)
            self.assertGreater(result["second_moe_counterexample_read_rms"], 0)
            self.assertGreater(result["second_moe_counterexample_expert_output_rms"], 0)


if __name__ == "__main__":
    unittest.main()
