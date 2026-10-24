"""Reference invariants; run with NANOGPT_PATH set."""

import unittest

import torch

from multistream import Config, MoE, StreamMix, initialized_model


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(vocab_size=65, block_size=8, n_layer=4, n_head=2,
                          n_embd=16, n_experts=2, expert_hidden=16, dense_hidden=16)

    def test_sinkhorn_and_mean(self):
        mix = StreamMix(4, 16)
        with torch.no_grad():
            mix.weight.copy_(0.1 * torch.randn_like(mix.weight))
        x = torch.randn(6, 4, 16)
        context = x.mean(1)
        matrix = mix.matrix(context)
        self.assertLess((matrix.sum(-1) - 1).abs().max().item(), 1e-4)
        self.assertLess((matrix.sum(-2) - 1).abs().max().item(), 1e-4)
        torch.testing.assert_close(mix(x, context).mean(1), x.mean(1), atol=1e-5, rtol=1e-5)

    def test_single_stream_lift_at_initialization(self):
        tokens = torch.randint(0, 65, (2, 8))
        a = initialized_model(self.cfg, "A")
        b = initialized_model(self.cfg, "B")
        torch.testing.assert_close(a(tokens)[0], b(tokens)[0], atol=2e-5, rtol=2e-5)

    def test_cloned_forward_and_asymmetric_streams(self):
        tokens = torch.randint(0, 65, (2, 8))
        models = [initialized_model(self.cfg, mode) for mode in "BCD"]
        outputs = [model(tokens) for model in models]
        for logits, loss, aux, streams in outputs[1:]:
            torch.testing.assert_close(logits, outputs[0][0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(aux, outputs[0][2], atol=1e-5, rtol=1e-5)
            self.assertGreater((streams[:, :, 0] - streams[:, :, 1]).abs().max().item(), 1e-5)

    def test_expert_read_write_and_router_gradients(self):
        torch.manual_seed(7)
        moe = MoE(self.cfg, "D")
        x = torch.randn(2, 3, 4, 16, requires_grad=True)
        y, aux = moe(x)
        (y.square().mean() + 0.01 * aux).backward()
        for p in (x, moe.read_weight, moe.write_weight, moe.mix.weight, moe.router.weight,
                  moe.experts[0].gate.weight, moe.experts[1].gate.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_second_moe_receives_differentiated_read_gradient(self):
        torch.manual_seed(13)
        model = initialized_model(self.cfg, "D")
        tokens = torch.randint(0, 65, (2, 8))
        targets = torch.randint(0, 65, (2, 8))
        model(tokens, targets)[1].backward()
        first = model.blocks[1].ff.read_weight.grad
        second = model.blocks[3].ff.read_weight.grad
        self.assertLess(first.abs().max().item(), 1e-9)
        self.assertGreater(second.abs().sum().item(), 1e-8)

    def test_input_gradient_against_finite_difference(self):
        torch.manual_seed(11)
        moe = MoE(self.cfg, "D").double()
        x = torch.randn(1, 1, 4, 16, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda t: moe(t)[0], (x,),
                                                  eps=1e-6, atol=3e-4, rtol=3e-3))

    def test_parameter_gradients_against_finite_difference(self):
        torch.manual_seed(11)
        moe = MoE(self.cfg, "D").double()
        x = torch.randn(1, 1, 4, 16, dtype=torch.double)
        def objective():
            output, aux = moe(x)
            return output.square().mean() + 0.01 * aux

        objective().backward()
        names = ("read_weight", "write_weight", "mix.weight", "router.weight",
                 "experts.0.gate.weight", "experts.1.up.weight")
        for name in names:
            parameter = dict(moe.named_parameters())[name]
            actual = parameter.grad.flatten()[0].item()
            with torch.no_grad():
                parameter.flatten()[0] += 1e-6
                plus = objective().item()
                parameter.flatten()[0] -= 2e-6
                minus = objective().item()
                parameter.flatten()[0] += 1e-6
            expected = (plus - minus) / 2e-6
            self.assertAlmostEqual(actual, expected, delta=2e-5 + 1e-3 * abs(expected), msg=name)

    def test_unselected_experts_are_not_dispatched(self):
        cfg = Config(n_embd=16, n_experts=4, expert_hidden=16)
        moe = MoE(cfg, "D")
        output, aux = moe(torch.randn(1, 1, 4, 16))
        (output.square().mean() + 0.01 * aux).backward()
        self.assertTrue(torch.isfinite(output).all())
        unused = sum(expert.gate.weight.grad is None for expert in moe.experts[1:])
        self.assertEqual(unused, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_bf16_against_fp32(self):
        model = initialized_model(self.cfg, "D").cuda()
        tokens = torch.randint(0, 65, (2, 8), device="cuda")
        targets = torch.randint(0, 65, (2, 8), device="cuda")
        full, full_loss, _, _ = model(tokens, targets)
        full_loss.backward()
        full_grad = model.blocks[3].ff.read_weight.grad.detach().clone()
        full_write_grad = model.blocks[3].ff.write_weight.grad.detach().clone()
        full_router_grad = model.blocks[3].ff.router.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mixed, mixed_loss, _, _ = model(tokens, targets)
        mixed_loss.backward()
        mixed_grad = model.blocks[3].ff.read_weight.grad.detach()
        self.assertLess((full - mixed.float()).abs().max().item(), 0.05)
        self.assertLess(abs(full_loss.item() - mixed_loss.item()), 0.03)
        self.assertLess((full_grad - mixed_grad).norm().item(), 1e-5)
        for exact, approximate in (
            (full_write_grad, model.blocks[3].ff.write_weight.grad),
            (full_router_grad, model.blocks[3].ff.router.weight.grad),
        ):
            self.assertGreater(exact.norm().item(), 1e-5)
            self.assertLess(((exact - approximate).norm() / exact.norm()).item(), 0.1)


if __name__ == "__main__":
    unittest.main()
