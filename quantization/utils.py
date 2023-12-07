# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from collections import namedtuple
from enum import Flag, auto
from functools import partial

import numpy as np


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)


class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""

    pass


MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())
