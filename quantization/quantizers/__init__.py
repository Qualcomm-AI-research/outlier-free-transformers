# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from quantization.quantizers.uniform_quantizers import (
    AsymmetricUniformQuantizer,
    SymmetricUniformQuantizer,
)
from quantization.utils import ClassEnumOptions, MethodMap


class QMethods(ClassEnumOptions):
    symmetric_uniform = MethodMap(SymmetricUniformQuantizer)
    asymmetric_uniform = MethodMap(AsymmetricUniformQuantizer)
