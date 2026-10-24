"""Qwen3-backed A/B/C/D warm-start reference, with no custom attention kernels."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

from multistream_moe.multistream import Config, MoE


class QwenExpert(nn.Module):
    def __init__(self, norm, mlp):
        super().__init__()
        self.norm = copy.deepcopy(norm)
        self.mlp = copy.deepcopy(mlp)
        with torch.no_grad():
            self.mlp.down_proj.weight.mul_(0.5)

    def forward(self, x):
        return self.mlp(self.norm(x))


class QwenMultiStream(nn.Module):
    """Owns a Qwen3ForCausalLM; alternating MLPs become source-side MoE layers."""

    def __init__(self, base, mode, n_experts=4, n_streams=4):
        super().__init__()
        if mode not in "ABCD" or n_experts < 2:
            raise ValueError("Mode A-D and at least two experts required")
        self.base = base
        self.mode = mode
        self.streams = 1 if mode == "A" else n_streams
        self.moes = nn.ModuleDict()
        for i, layer in enumerate(base.model.layers):
            if i % 2 == 1:
                experts = [QwenExpert(layer.post_attention_layernorm, layer.mlp)
                           for _ in range(n_experts + 1)]
                cfg = Config(vocab_size=base.config.vocab_size,
                             block_size=base.config.max_position_embeddings,
                             n_layer=base.config.num_hidden_layers,
                             n_head=base.config.num_attention_heads,
                             n_embd=base.config.hidden_size, n_experts=n_experts,
                             expert_hidden=base.config.intermediate_size,
                             dense_hidden=base.config.intermediate_size,
                             n_streams=n_streams)
                self.moes[str(i)] = MoE(cfg, mode, experts=experts)
                layer.post_attention_layernorm = nn.Identity()
                layer.mlp = nn.Identity()

    def forward(self, tokens, targets=None):
        batch, seq = tokens.shape
        if seq > self.base.config.max_position_embeddings:
            raise ValueError("sequence exceeds Qwen context")
        x = self.base.model.embed_tokens(tokens)
        position_ids = torch.arange(seq, device=tokens.device).unsqueeze(0)
        positions = self.base.model.rotary_emb(x, position_ids)
        mask = create_causal_mask(config=self.base.config, inputs_embeds=x,
                                  attention_mask=None, past_key_values=None,
                                  position_ids=position_ids)
        x = x.unsqueeze(2).expand(batch, seq, self.streams, -1)
        aux = x.new_zeros(())
        for i, layer in enumerate(self.base.model.layers):
            mean = x.mean(dim=2)
            attn, _ = layer.self_attn(layer.input_layernorm(mean),
                                      position_embeddings=positions, attention_mask=mask)
            x = x + attn.unsqueeze(2)
            if str(i) in self.moes:
                x, block_aux = self.moes[str(i)](x)
                aux = aux + block_aux
            else:
                mean = x.mean(dim=2)
                update = layer.mlp(layer.post_attention_layernorm(mean))
                x = x + update.unsqueeze(2)
        logits = self.base.lm_head(self.base.model.norm(x.mean(dim=2)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
        )
        return logits, loss, aux, x
