# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from torch.autograd import Function


class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class ScaleGradient(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad * ctx.scale, None


round_ste_func = RoundStraightThrough.apply
scale_grad_func = ScaleGradient.apply


class QuantizerNotInitializedError(Exception):
    """Raised when a quantizer has not initialized"""

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__(
            "Quantizer has  not been initialized yet"
        )
