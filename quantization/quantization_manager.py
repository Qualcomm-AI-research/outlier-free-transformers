# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from torch import nn

from quantization.qstates import Qstates
from quantization.quantizers import QMethods
from quantization.quantizers.base_quantizers import QuantizerBase
from quantization.quantizers.quantizer_utils import QuantizerNotInitializedError
from quantization.range_estimators import RangeEstimatorBase, RangeEstimators


class QuantizationManager(nn.Module):
    """Implementation of Quantization and Quantization Range Estimation

    Parameters
    ----------
    n_bits: int
        Number of bits for the quantization.
    qmethod: QMethods member (Enum)
        The quantization scheme to use, e.g. symmetric_uniform, asymmetric_uniform,
        qmn_uniform etc.
    init: RangeEstimators member (Enum)
        Initialization method for the grid from
    per_channel: bool
        If true, will use a separate quantization grid for each kernel/channel.
    x_min: float or PyTorch Tensor
        The minimum value which needs to be represented.
    x_max: float or PyTorch Tensor
        The maximum value which needs to be represented.
    """

    def __init__(
        self,
        qmethod: QuantizerBase = QMethods.symmetric_uniform.cls,
        init: RangeEstimatorBase = RangeEstimators.current_minmax.cls,
        per_channel=False,
        x_min=None,
        x_max=None,
        qparams=None,
        init_params=None,
    ):
        super().__init__()
        self.state = Qstates.estimate_ranges
        self.qmethod = qmethod
        self.init = init
        self.per_channel = per_channel
        self.qparams = qparams if qparams else {}
        self.init_params = init_params if init_params else {}
        self.range_estimator = None

        # define quantizer
        self.quantizer = self.qmethod(per_channel=self.per_channel, **qparams)
        self.quantizer.state = self.state

        # define range estimation method for quantizer initialisation
        if x_min is not None and x_max is not None:
            self.set_quant_range(x_min, x_max)
            self.fix_ranges()
        else:
            # set up the collector function to set the ranges
            self.range_estimator = self.init(
                per_channel=self.per_channel, quantizer=self.quantizer, **self.init_params
            )

    @property
    def n_bits(self):
        return self.quantizer.n_bits

    def estimate_ranges(self):
        self.state = Qstates.estimate_ranges
        self.quantizer.state = self.state

    def fix_ranges(self):
        if self.quantizer.is_initialized:
            self.state = Qstates.fix_ranges
            self.quantizer.state = self.state
            self.quantizer.fix_ranges()
        else:
            raise QuantizerNotInitializedError()

    def learn_ranges(self):
        self.quantizer.make_range_trainable()
        self.state = Qstates.learn_ranges
        self.quantizer.state = self.state

    def estimate_ranges_train(self):
        self.state = Qstates.estimate_ranges_train
        self.quantizer.state = self.state

    def reset_ranges(self):
        self.range_estimator.reset()
        self.quantizer.reset()
        self.estimate_ranges()

    def forward(self, x):
        if self.state == Qstates.estimate_ranges or (
            self.state == Qstates.estimate_ranges_train and self.training
        ):
            # Note this can be per tensor or per channel
            cur_xmin, cur_xmax = self.range_estimator(x)
            self.set_quant_range(cur_xmin, cur_xmax)

        return self.quantizer(x)

    def set_quant_range(self, x_min, x_max):
        self.quantizer.set_quant_range(x_min, x_max)

    def extra_repr(self):
        return "state={}".format(self.state.name)
