# coding=utf-8
# Copyright @lucidrains
# ref. https://github.com/lucidrains/rotary-embedding-torch

from inspect import isfunction

import torch
from torch import nn, einsum
from einops import rearrange, repeat


def flip_every_two(t):
    t = rearrange(t, "b (n r) ... -> b n r ...", r=2)
    # so we pay attention to the off-diagonal blocks in the attention matrix
    t = torch.flip(t, dims=(2,))
    t = rearrange(t, "b n r ... -> b (n r) ...")
    return t


class RotaryEmbedding(nn.Module):
    def __init__(self, theta: int, dim: int, learned_freq: bool):
        super().__init__()
        self.theta = theta
        self.dim = dim
        self.learned_freq = learned_freq
        freqs = 1.0 / (theta ** torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        self.cache = dict()

        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, t, cache_key=None):
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if cache_key is not None:
            self.cache[cache_key] = freqs

        return freqs

    @staticmethod
    def apply_rotary_emb(freqs, t, start_index=0):
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f"feature dimension {t.shape[-1]} is not of sufficient "
            f"size to rotate in all the positions {rot_dim}"
        )
        t_left = t[..., :start_index]
        t = t[..., start_index:end_index]
        t_right = t[..., end_index:]

        def rotary_half(x):
            x = rearrange(x, "... (d r) -> ... d r", r=2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return rearrange(x, "... d r -> ... (d r)")

        t = (t * freqs.cos()) + (rotary_half(t) * freqs.sin())
        return torch.cat((t_left, t, t_right), dim=-1)
