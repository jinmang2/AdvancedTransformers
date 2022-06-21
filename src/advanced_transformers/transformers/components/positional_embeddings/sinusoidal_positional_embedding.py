# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.

import math
from typing import Any, Optional

import torch
from torch import Tensor, nn


class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.
    We don't want to save the weight of this embedding since it's not trained (deterministic)
    and it can be huge. Padding symbols are ignored.
    These embeddings get automatically extended in forward if more positions is needed.
    """

    def __init__(self, num_positions, embedding_dim, padding_idx):
        self.make_weight(num_positions, embedding_dim, padding_idx)

    def make_weight(self, num_positions, embedding_dim, padding_idx):
        weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
        if not hasattr(self, "weight"):
            # in ___init__
            super().__init__(num_positions, embedding_dim, padding_idx, _weight=weight)
        else:
            # in forward put the weights on the correct dtype and device of the param
            weight = weight.to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight = nn.Parameter(weight)
        self.weight.detach_()
        self.weight.requires_grad = False

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        """
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor,
        but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """
        Replace non-padding symbols with their position numbers.
        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len

        if max_pos > self.weight.size(0):
            # expand embeddings if needed
            self.make_weight(max_pos, self.embedding_dim, self.padding_idx)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weight[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = self.make_positions(input, self.padding_idx)
        # `super().forward` is
        # (self.weight.index_select(0, positions.view(-1))
        #  .view(bsz, seq_len, -1).detach())
        return super().forward(positions)
