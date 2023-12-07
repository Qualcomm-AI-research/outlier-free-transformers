# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from enum import auto

from quantization.utils import BaseEnumOptions


class Qstates(BaseEnumOptions):
    estimate_ranges = auto()  # ranges are updated in eval and train mode
    fix_ranges = auto()  # quantization ranges are fixed for train and eval
    learn_ranges = auto()  # quantization params are nn.Parameters
    estimate_ranges_train = (
        auto()
    )  # quantization ranges are updated during train and fixed for eval
