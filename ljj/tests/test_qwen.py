"""Small, offline Qwen3 configuration checks before using pretrained weights."""

import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from multistream_moe.qwen_reference import QwenMultiStream
from experiments.train_qwen import forward_batch, learning_rate, mechanism_metrics


class QwenReferenceTests(unittest.TestCase):
    def test_learning_rate_warmup_and_legacy_schedule(self):
        self.assertAlmostEqual(learning_rate(1e-5, 0, 1000, 100), 1e-7)
        self.assertAlmostEqual(learning_rate(1e-5, 99, 1000, 100), 1e-5)
        self.assertAlmostEqual(learning_rate(1e-5, 100, 1000, 100), 1e-5)
        self.assertAlmostEqual(learning_rate(1e-5, 999, 1000, 100), 1e-6)
        self.assertAlmostEqual(learning_rate(1e-5, 0, 32, 0), 1e-5)
        self.assertAlmostEqual(learning_rate(1e-5, 31, 32, 0), 1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA mechanism probe")
    def test_mechanism_metrics_starts_with_zero_read_diversity(self):
        torch.manual_seed(42)
        model = QwenMultiStream(copy.deepcopy(self.base), "D").cuda().eval()
        metrics = mechanism_metrics(model, "D", self.tokens.cuda())
        self.assertEqual(metrics["second_read_expert_weight_rms"], 0)
        self.assertEqual(metrics["second_read_expert_bias_rms"], 0)
        self.assertEqual(metrics["second_write_expert_weight_rms"], 0)
        self.assertGreater(metrics["final_stream_centered_rms"], 0)

    def setUp(self):
        torch.manual_seed(9)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8,
                          max_position_embeddings=32, tie_word_embeddings=True)
        cfg._attn_implementation = "sdpa"
        self.base = Qwen3ForCausalLM(cfg).eval()
        self.tokens = torch.randint(0, 128, (2, 8))

    def test_pretrained_lift_and_matched_variants(self):
        with torch.no_grad():
            expected = self.base(self.tokens).logits
        outputs = []
        states = []
        routers = []
        for mode in "ABCD":
            torch.manual_seed(42)
            model = QwenMultiStream(copy.deepcopy(self.base), mode).eval()
            with torch.no_grad():
                logits, _, _, streams = model(self.tokens)
            torch.testing.assert_close(logits, expected, atol=2e-4, rtol=2e-4)
            outputs.append(logits)
            states.append(streams)
            routers.append(model.moes["1"].router.weight.detach())
            if mode != "A":
                self.assertGreater((streams[:, :, 0] - streams[:, :, 1]).abs().max().item(), 0)
        for logits in outputs[1:]:
            torch.testing.assert_close(logits, outputs[0], atol=2e-4, rtol=2e-4)
        for state in states[2:]:
            torch.testing.assert_close(state, states[1], atol=1e-5, rtol=1e-5)
        for router in routers[1:]:
            torch.testing.assert_close(router, routers[0], atol=0, rtol=0)

    def test_causal_mask_and_independent_read_gradient(self):
        torch.manual_seed(42)
        model = QwenMultiStream(copy.deepcopy(self.base), "D").eval()
        future = self.tokens.clone()
        future[:, -1] = (future[:, -1] + 1) % 128
        with torch.no_grad():
            original = model(self.tokens)[0]
            altered = model(future)[0]
        torch.testing.assert_close(original[:, :-1], altered[:, :-1], atol=1e-5, rtol=1e-5)
        labels = torch.randint(0, 128, self.tokens.shape)
        model(self.tokens, labels)[1].backward()
        first = model.moes["1"].read_weight.grad
        second = model.moes["3"].read_weight.grad
        self.assertLess(first.abs().max().item(), 1e-9)
        self.assertGreater(second.abs().sum().item(), 1e-8)

    def test_dense_teacher_forced_loss_uses_next_token_targets_once(self):
        target = torch.roll(self.tokens, shifts=-1, dims=1)
        logits, loss, aux = forward_batch(self.base, "dense", self.tokens, target)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
        )
        torch.testing.assert_close(loss, expected)
        self.assertEqual(aux.item(), 0)


if __name__ == "__main__":
    unittest.main()
