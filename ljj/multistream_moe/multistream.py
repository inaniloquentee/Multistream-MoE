"""Reference model: per-token stream mixing and expert-specific reads/writes."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def nano_gpt_layers():
    upstream = Path(os.environ.get("NANOGPT_PATH", "")).expanduser()
    if not os.environ.get("NANOGPT_PATH") or not (upstream / "model.py").is_file():
        raise RuntimeError("Set NANOGPT_PATH to the pinned nanoGPT checkout for the scratch model")
    sys.path.insert(0, str(upstream.resolve()))
    from model import CausalSelfAttention, GPTConfig, LayerNorm
    return CausalSelfAttention, GPTConfig, LayerNorm


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_experts: int = 4
    expert_hidden: int = 128
    dense_hidden: int = 128
    n_streams: int = 4
    top_k: int = 2


class SwiGLU(nn.Module):
    def __init__(self, width, hidden, internal_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(width) if internal_norm else nn.Identity()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StreamMix(nn.Module):
    """Token-conditional Sinkhorn matrix; identity for the single-stream baseline."""

    def __init__(self, streams, width):
        super().__init__()
        self.streams = streams
        self.weight = nn.Parameter(torch.zeros(streams * streams, width))
        self.bias = nn.Parameter(torch.zeros(streams * streams))

    def matrix(self, context):
        if self.streams == 1:
            return context.new_ones(context.shape[0], 1, 1)
        dtype = torch.float32 if context.dtype in (torch.float16, torch.bfloat16) else context.dtype
        score = F.linear(context.to(dtype), self.weight.to(dtype), self.bias.to(dtype))
        q = score.view(-1, self.streams, self.streams).softmax(dim=-1)
        for _ in range(20):
            q = q / q.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-2, keepdim=True)
        eye = torch.eye(self.streams, device=q.device, dtype=q.dtype)
        return 0.95 * eye + 0.05 * q

    def forward(self, x, context):
        r = self.matrix(context).to(x.dtype)
        return torch.einsum("nij,njd->nid", r, x)


class MoE(nn.Module):
    def __init__(self, cfg, mode, experts=None):
        super().__init__()
        if mode not in "ABCD":
            raise ValueError(mode)
        self.mode = mode
        self.streams = 1 if mode == "A" else cfg.n_streams
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.mix = StreamMix(self.streams, cfg.n_embd)
        self.context_norm = nn.LayerNorm(cfg.n_embd, bias=False)
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(experts if experts is not None else [
            SwiGLU(cfg.n_embd, cfg.expert_hidden) for _ in range(cfg.n_experts + 1)
        ])
        if len(self.experts) != cfg.n_experts + 1:
            raise ValueError("Need one shared and n_experts routed experts")
        if mode != "A":
            read_maps = cfg.n_experts + 1 if mode == "D" else 1
            write_maps = cfg.n_experts + 1 if mode in "CD" else 1
            self.read_weight = nn.Parameter(torch.zeros(read_maps, self.streams, cfg.n_embd))
            self.read_bias = nn.Parameter(torch.zeros(read_maps, self.streams))
            self.write_weight = nn.Parameter(torch.zeros(write_maps, self.streams, cfg.n_embd))
            init_write = torch.tensor([1.03, 0.99, 0.99, 0.99])
            self.write_bias = nn.Parameter(torch.logit(init_write / 2).repeat(write_maps, 1))

    def read(self, x, context, expert):
        if self.mode == "A":
            return x[:, 0, :]
        row = expert if self.mode == "D" else 0
        a = F.linear(context, self.read_weight[row], self.read_bias[row]).softmax(-1)
        return torch.einsum("nsd,ns->nd", x, a.to(x.dtype))

    def write(self, context, expert):
        if self.mode == "A":
            return context.new_ones(context.shape[0], 1)
        row = expert if self.mode in "CD" else 0
        return 2 * torch.sigmoid(F.linear(
            context, self.write_weight[row], self.write_bias[row]
        ))

    def forward(self, x):
        batch, seq, streams, width = x.shape
        flat = x.reshape(-1, streams, width)
        context = self.context_norm(flat.mean(dim=1))
        logits = self.router(context)
        top_logits, indices = logits.topk(self.top_k, dim=-1)
        gates = top_logits.softmax(dim=-1)
        shared = self.experts[0](self.read(flat, context, 0))
        update = shared.unsqueeze(1) * self.write(context, 0).to(x.dtype)[:, :, None]
        routed = torch.zeros_like(flat)
        for j in range(self.n_experts):
            token, slot = torch.where(indices == j)
            if token.numel():
                expert = j + 1
                state = flat.index_select(0, token)
                ctx = context.index_select(0, token)
                z = self.experts[expert](self.read(state, ctx, expert))
                contribution = (gates[token, slot, None, None] * z[:, None, :]
                                * self.write(ctx, expert).to(x.dtype)[:, :, None])
                routed = routed.index_add(0, token, contribution)
        probs = logits.softmax(dim=-1)
        load = torch.bincount(indices.flatten(), minlength=self.n_experts).to(logits.dtype)
        aux = self.n_experts * ((load / indices.numel()) * probs.mean(dim=0)).sum()
        result = self.mix(flat, context) + update + routed
        return result.reshape(batch, seq, streams, width), aux


class Block(nn.Module):
    """Attention/dense updates read the mean; residual states stay separate."""

    def __init__(self, cfg, mode, index):
        super().__init__()
        CausalSelfAttention, GPTConfig, LayerNorm = nano_gpt_layers()
        nano_cfg = GPTConfig(block_size=cfg.block_size, n_head=cfg.n_head,
                             n_embd=cfg.n_embd, dropout=0.0, bias=False)
        self.ln_attn = LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(nano_cfg)
        self.ln_dense = LayerNorm(cfg.n_embd, bias=False)
        self.ff = MoE(cfg, mode) if index % 2 == 1 else SwiGLU(
            cfg.n_embd, cfg.dense_hidden, internal_norm=False
        )

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x.mean(dim=2))).unsqueeze(2)
        if isinstance(self.ff, MoE):
            return self.ff(x)
        update = self.ff(self.ln_dense(x.mean(dim=2))).unsqueeze(2)
        return x + update, x.new_zeros(())


class Model(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        if cfg.n_layer < 4 or cfg.top_k > cfg.n_experts or cfg.n_streams != 4:
            raise ValueError("Need >=4 layers (2 MoEs), TopK <= experts and four streams")
        self.cfg, self.mode = cfg, mode
        self.streams = 1 if mode == "A" else cfg.n_streams
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, mode, i) for i in range(cfg.n_layer))
        _, _, LayerNorm = nano_gpt_layers()
        self.ln_f = LayerNorm(cfg.n_embd, bias=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        batch, seq = tokens.shape
        if seq > self.cfg.block_size:
            raise ValueError("sequence exceeds block size")
        x = self.wte(tokens) + self.wpe(torch.arange(seq, device=tokens.device))
        x = x.unsqueeze(2).expand(batch, seq, self.streams, -1)
        aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x)
            aux = aux + block_aux
        logits = self.lm_head(self.ln_f(x.mean(dim=2)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
        )
        return logits, loss, aux, x


def initialized_model(cfg, mode, seed=42):
    """Clone every shared B tensor into C/D, broadcasting new expert map rows."""
    torch.manual_seed(seed)
    if mode in "AB":
        return Model(cfg, mode)
    reference = Model(cfg, "B")
    torch.manual_seed(seed)
    candidate = Model(cfg, mode)
    source = reference.state_dict()
    target = candidate.state_dict()
    for key, value in target.items():
        if key in source:
            if source[key].shape == value.shape:
                value.copy_(source[key])
            elif key.endswith(("read_weight", "write_weight", "read_bias", "write_bias")):
                value.copy_(source[key].expand_as(value))
            else:
                raise ValueError(f"Unexpected parameter shape: {key}")
    candidate.load_state_dict(target)
    return candidate
