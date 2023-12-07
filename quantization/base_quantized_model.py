# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import warnings
from typing import Dict, Union

import torch
from torch import Tensor, nn

from quantization.base_quantized_classes import (
    QuantizedModule,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
    _set_layer_fix_ranges,
    _set_layer_learn_ranges,
)
from quantization.quantizers.base_quantizers import QuantizerBase


class QuantizedModel(nn.Module):
    """
    Parent class for a quantized model. This allows you to have convenience functions to put the
    whole model into quantization or full precision or to freeze BN. Otherwise it does not add any
    further functionality, so it is not a necessity that a quantized model uses this class.
    """

    def __init__(self, input_size=(1, 3, 224, 224)):
        """
        Parameters
        ----------
        input_size:     Tuple with the input dimension for the model (including batch dimension)
        """
        super().__init__()
        self.input_size = input_size

    def load_state_dict(
        self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True
    ):
        """
        This function overwrites the load_state_dict of nn.Module to ensure that quantization
        parameters are loaded correctly for quantized model.

        """
        quant_state_dict = {
            k: v for k, v in state_dict.items() if k.endswith("_quant_a") or k.endswith("_quant_w")
        }

        if quant_state_dict:
            # Case 1: the quantization states are stored in the state_dict
            super().load_state_dict(quant_state_dict, strict=False)

        else:
            # Case 2 (older models): the quantization states are NOT stored in the state_dict but
            # only the scale factor _delta.
            warnings.warn(
                "Old state_dict without quantization state included. Checking for " "_delta instead"
            )
            # Add quantization flags to the state_dict
            for name, module in self.named_modules():
                if isinstance(module, QuantizedModule):
                    state_dict[".".join((name, "_quant_a"))] = torch.BoolTensor([False])
                    state_dict[".".join((name, "_quant_w"))] = torch.BoolTensor([False])
                    if (
                        ".".join((name, "activation_quantizer", "quantizer", "_delta"))
                        in state_dict.keys()
                    ):
                        module.quantized_acts()
                        state_dict[".".join((name, "_quant_a"))] = torch.BoolTensor([True])
                    if (
                        ".".join((name, "weight_quantizer", "quantizer", "_delta"))
                        in state_dict.keys()
                    ):
                        module.quantized_weights()
                        state_dict[".".join((name, "_quant_w"))] = torch.BoolTensor([True])

        # Pass dummy data through quantized model to ensure all quantization parameters are
        # initialized with the correct dimensions (None tensors will lead to issues in state dict loading)
        device = next(self.parameters()).device
        dummy_input = torch.rand(*self.input_size, device=device)
        with torch.no_grad():
            self.forward(dummy_input)

        # Load state dict
        super().load_state_dict(state_dict, strict)

    def disable_caching(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.disable_caching()

        self.apply(_fn)

    def quantized_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_weights()

        self.apply(_fn)

    def quantized_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def quantized(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized()

        self.apply(_fn)

    def full_precision(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision()

        self.apply(_fn)

    # Methods for switching quantizer quantization states
    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def set_quant_state(self, weight_quant, act_quant):
        if act_quant:
            self.quantized_acts()
        else:
            self.full_precision_acts()

        if weight_quant:
            self.quantized_weights()
        else:
            self.full_precision_weights()

    def grad_scaling(self, grad_scaling=True):
        def _fn(module):
            if isinstance(module, QuantizerBase):
                module.grad_scaling = grad_scaling

        self.apply(_fn)
