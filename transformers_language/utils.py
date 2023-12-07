# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn

from quantization.range_estimators import RangeEstimators
from quantization.utils import StopForwardException


def kurtosis(x, eps=1e-6):
    """x - (B, d)"""
    mu = x.mean(dim=1, keepdims=True)
    s = x.std(dim=1)
    mu4 = ((x - mu) ** 4.0).mean(dim=1)
    k = mu4 / (s**4.0 + eps)
    return k


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))


def val_qparams(config):
    weight_range_options = {}
    if config.quant.weight_quant_method == RangeEstimators.MSE:
        weight_range_options = dict(opt_method=config.quant.weight_opt_method)
    if config.quant.num_candidates is not None:
        weight_range_options["num_candidates"] = config.quant.num_candidates

    params = {
        "method": config.quant.qmethod.cls,
        "n_bits": config.quant.n_bits,
        "n_bits_act": config.quant.n_bits_act,
        "act_method": config.quant.qmethod_act.cls,
        "per_channel_weights": config.quant.per_channel,
        "percentile": config.quant.percentile,
        "quant_setup": config.quant.quant_setup,
        "weight_range_method": config.quant.weight_quant_method.cls,
        "weight_range_options": weight_range_options,
        "act_range_method": config.act_quant.quant_method.cls,
        "act_range_options": config.act_quant.options,
    }
    return params


def pass_data_for_range_estimation(loader, model, act_quant=None, max_num_batches=20, inp_idx=0):
    model.eval()
    batches = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, data in enumerate(loader):
            try:
                if isinstance(data, (tuple, list)):
                    x = data[inp_idx].to(device=device)
                    batches.append(x.data.cpu().numpy())
                    model(x)
                    print(f"proccesed step={i}")
                else:
                    x = {k: v.to(device=device) for k, v in data.items()}
                    model(**x)
                    print(f"proccesed step={i}")

                if i >= max_num_batches - 1 or not act_quant:
                    break
            except StopForwardException:
                pass
        return batches


class DotDict(dict):
    """
    This class enables access to its attributes as both ['attr'] and .attr .
    Its advantage is that content of its `instance` can be accessed with `.`
    and still passed to functions as `**instance` (as dictionaries) for
    implementing variable-length arguments.
    """

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")
