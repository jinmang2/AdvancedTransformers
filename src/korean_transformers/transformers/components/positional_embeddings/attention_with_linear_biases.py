# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import List

import torch


def get_slopes(n: int) -> List[int]:
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    # In the paper, we only train models that have 2^a heads for some a.
    # This function has some good properties that only occur when the input is a power of 2.
    # To maintain that even when the number of heads is not a power of 2, we use this workaround.
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


if __name__ == "__main__":
    # __init__
    bsz = 32
    seq_len = 17
    max_tokens = 512
    maxpos = 512  # tokens_per_sample
    attn_heads = 16  # decoder_attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))
    # In the next line, the part after the * is what constructs the diagonal matrix
    # (right matrix in Figure 3 in the paper).
    # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3,
    # but one where all rows are identical.
    # This works because the softmax operation is invariant to translation,
    # and our bias functions are always linear.
    m = slopes.unsqueeze(1).unsqueeze(1)  # head-specific slope fixed
    positions = (
        torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
    )
    alibi = m * positions  # non-learned bias
    alibi = alibi.view(attn_heads, 1, maxpos)
    alibi = alibi.repeat(max_tokens // maxpos, 1, 1)  # batch_size, 1, 1
    # extract_features_scriptable
    # we move the mask construction `before layer operation` because its slightly more efficient
    # self_attn_mask = self.buffered_future_mask(x)

    def fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    _future_mask = torch.triu(fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask + alibi
    _future_mask = _future_mask[: bsz * attn_heads, :seq_len, :seq_len]
