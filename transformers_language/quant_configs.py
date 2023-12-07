# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from quantization.quantizers import QMethods
from quantization.range_estimators import RangeEstimators
from transformers_language.utils import DotDict


def get_quant_config():
    config = DotDict()
    config.act_quant = DotDict(
        {
            "cross_entropy_layer": None,
            "num_batches": 16,
            "options": {},
            "quant_method": RangeEstimators.running_minmax,
            "std_dev": None,
        }
    )
    config.quant = DotDict(
        {
            "act_quant": True,
            "n_bits": 8,
            "n_bits_act": 8,
            "num_candidates": None,
            "per_channel": False,
            "percentile": None,
            "quant_setup": "all",
            "qmethod": QMethods.symmetric_uniform,
            "qmethod_act": QMethods.asymmetric_uniform,
            "weight_quant": True,
            "weight_quant_method": RangeEstimators.current_minmax,
        }
    )
    return config
