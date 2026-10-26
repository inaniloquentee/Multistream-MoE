"""The opt-in MoE read path accumulates in FP32 before its output cast."""

import unittest

import torch
from torch.nn import functional as F

from multistream_moe.multistream import Config, MoE


class ReadPrecisionTests(unittest.TestCase):
    def test_fp32_read_matches_explicit_accumulation(self):
        torch.manual_seed(7)
        cfg = Config(n_embd=32, n_experts=4)
        moe = MoE(cfg, "D")
        x = torch.randn(7, 4, 32).to(torch.bfloat16)
        context = torch.randn(7, 32).to(torch.bfloat16)
        with torch.no_grad():
            moe.read_bias[1].copy_(torch.tensor([0.75, -0.25, -0.25, -0.25]))
        self.assertFalse(moe.fp32_read)
        moe.fp32_read = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = moe.read(x, context, 1)
        weights = F.linear(context.float(), moe.read_weight[1].float(),
                           moe.read_bias[1].float()).softmax(-1)
        expected = torch.einsum("nsd,ns->nd", x.float(), weights).to(x.dtype)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
