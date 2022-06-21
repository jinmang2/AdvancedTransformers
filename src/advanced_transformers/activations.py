# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import torch
from packaging import version
from torch import Tensor, nn

from .quant_modules import (
    symmetric_linear_quantization_params,
    SymmetricQuantFunction,
    floor_ste,
    FixedPointMul,
)


# Copied from transformers.activations.NewGELUActivation
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# Copied from transformers.activations.GELUActivation
class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# Copied from transformers.activations.FastGELUActivation
class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate.
    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


# Copied from transformers.activations.QuickGELUActivation
class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate.
    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


# Copied from transformers.activations.ClippedGELUActivation
class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


# Copied from transformers.models.bloom.modeling_bloom.bloom_gelu_forward
def bloom_gelu_forward(x):
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.
    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# Copied from transformers.models.bloom.modeling_bloom.bloom_gelu_back
def bloom_gelu_back(g, x):
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)
    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


# Copied from transformers.models.bloom.modeling_bloom.GeLUFunction
class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


# Copied from transformers.models.bloom.modeling_bloom.BloomGelu
class BloomGELUActivation(nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs
    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            return GeLUFunction.apply(x)
        else:
            bloom_gelu_forward(x)


# Inspired by transformers.models.ibert.quant_modules.IntGELU
class IntGELU(nn.Module):
    """
    Quantized version of `torch.nn.GELU`. Adds quantization-specific arguments on top of `torch.nn.GELU`.
    Args:
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, quant_mode=True):
        super().__init__()
        self.quant_mode = quant_mode

        if not self.quant_mode:
            self.activation_fn = nn.GELU()

        self.k = 1.4142
        self.const = 14  # dummy integer constant
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def int_erf(self, x_int, scaling_factor):
        b_int = torch.floor(self.coeff[1] / scaling_factor)
        c_int = torch.floor(self.coeff[2] / scaling_factor**2)
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor**2 * self.coeff[0]

        # avoid overflow
        y_int = floor_ste.apply(y_int / 2**self.const)
        scaling_factor = scaling_factor * 2**self.const

        return y_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            return self.activation_fn(x), None

        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = 1.0 // sigmoid_scaling_factor

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor


# Copied from transformers.models.ibert.quant_modules.QuantAct
class QuantAct(nn.Module):
    """
    Quantizes the given activation.
    Args:
        activation_bit (`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.
        channel_len (`int`, *optional*):
            Specify the channel length when set the *per_channel* True.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(
        self,
        activation_bit,
        act_range_momentum=0.95,
        per_channel=False,
        channel_len=None,
        quant_mode=False
    ):
        super().__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.percentile = False
        self.act_function = SymmetricQuantFunction.apply

        if not self.per_channel:
            self.register_buffer("x_min", torch.zeros(1))
            self.register_buffer("x_max", torch.zeros(1))
            self.register_buffer("act_scaling_factor", torch.zeros(1))
            self.x_min -= 1e-5
            self.x_max += 1e-5
        else:
            raise NotImplementedError("per-channel mode is not currently supported for activation.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(activation_bit={self.activation_bit}, "
            f"quant_mode: {self.quant_mode}, Act_min: {self.x_min.item():.2f}, "
            f"Act_max: {self.x_max.item():.2f})"
        )

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        specified_min=None,
        specified_max=None,
    ):

        x_act = x if identity is None else identity + x
        # collect running stats if training
        if self.training:
            assert not self.percentile, "percentile mode is not currently supported for activation."
            assert not self.per_channel, "per-channel mode is not currently supported for activation."
            x_min = x_act.data.min()
            x_max = x_act.data.max()

            assert (
                x_max.isnan().sum() == 0 and x_min.isnan().sum() == 0
            ), "NaN detected when computing min/max of the activation"

            # Initialization
            if self.x_min.min() > -1.1e-5 and self.x_max.max() < 1.1e-5:
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        if not self.quant_mode:
            return x_act, None

        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, per_channel=self.per_channel
        )

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            quant_act_int = self.act_function(x, self.activation_bit, self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = FixedPointMul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


# Copied from transformers.activations.SiLUActivation
class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.7"):
            self.act = self._silu_python
        else:
            self.act = nn.functional.silu

    def _silu_python(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# Copied from transformers.activations.MishActivation
class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# Copied from transformers.activations.LinearActivation
class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


ACT2FN = {
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
    "gelu_fast": FastGELUActivation(),
    "gelu_new": NewGELUActivation(),
    "gelu_python": GELUActivation(use_gelu_python=True),
    "linear": LinearActivation(),
    "mish": MishActivation(),
    "quick_gelu": QuickGELUActivation(),
    "gelu_bloom": BloomGELUActivation(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": SiLUActivation(),
    "swish": SiLUActivation(),
    "tanh": nn.Tanh(),
    "gelu_int": IntGELU(quant_mode=False),
    "gelu_int_quant": IntGELU(),
    "quant_act_8": QuantAct(activation_bit=8),
    "quant_act_8_quant": QuantAct(activation_bit=8, quant_mode=True),
    "quant_act_16": QuantAct(activation_bit=16),
    "quant_act_16_quant": QuantAct(activation_bit=16, quant_mode=True),
    "quant_act_22": QuantAct(activation_bit=22),
    "quant_act_22_quant": QuantAct(activation_bit=22, quant_mode=True),
}