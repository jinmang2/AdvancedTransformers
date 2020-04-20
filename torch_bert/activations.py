# https://subinium.github.io/introduction-to-activation/

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def swish(x):
    """https://arxiv.org/pdf/1710.05941v1.pdf"""
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    # torch.erf(input, out=None) -> Tensor
    # Computes the error function of each element.
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu # 얘가 제일 빠름!
    try:
        import torch_xla

        logger.warning(
            "The torch_xla package was detected in the python environment. PyTorch/XLA and JIT is untested,"
            " no activation function will be traced with JIT."
        )
    except ImportError:
        gelu_new = torch.jit.script(gelu_new)

ACT2FN = {
    'relu': F.relu,
    'swish': swish,
    'gelu': gelu,
    'tanh': torch.tanh,
    'gelu_new': gelu_new
}


def get_activation(activation_string):
    activation = ACT2FN.get(activation_string, None)
    if activation is None:
        raise KeyError(f"function {activation_string} not found "
                        "in ACT2FN mapping {list(ACT2FN.keys())}")
    return activation
