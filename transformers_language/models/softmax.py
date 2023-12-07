# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from functools import partial

import torch


def clipped_softmax(data, dim=1, eta=1.1, gamma=-0.1, **kw):
    sm_out = torch.nn.functional.softmax(data, dim=dim, **kw)
    stretched_out = sm_out * (eta - gamma) + gamma
    return torch.clip(stretched_out, 0, 1)


SOFTMAX_MAPPING = {
    "vanilla": torch.nn.functional.softmax,
    # Clipped softmax
    "clipped(0:1.0003)": partial(clipped_softmax, gamma=0, eta=1.0003),
    "clipped(0:1.001)": partial(clipped_softmax, gamma=0, eta=1.001),
    "clipped(0:1.002)": partial(clipped_softmax, gamma=0, eta=1.002),
    "clipped(0:1.003)": partial(clipped_softmax, gamma=0, eta=1.003),
    "clipped(0:1.004)": partial(clipped_softmax, gamma=0, eta=1.004),
    "clipped(0:1.01)": partial(clipped_softmax, gamma=0, eta=1.01),
    "clipped(0:1.02)": partial(clipped_softmax, gamma=0, eta=1.02),
    "clipped(0:1.03)": partial(clipped_softmax, gamma=0, eta=1.03),
    "clipped(0:1.1)": partial(clipped_softmax, gamma=0, eta=1.1),
    "clipped(-.1:1)": partial(clipped_softmax, gamma=-0.1, eta=1.0),
    "clipped(-.00001:1)": partial(clipped_softmax, gamma=-0.00001, eta=1.0),
    "clipped(-.00003:1)": partial(clipped_softmax, gamma=-0.00003, eta=1.0),
    "clipped(-.0001:1)": partial(clipped_softmax, gamma=-0.0001, eta=1.0),
    "clipped(-.0003:1)": partial(clipped_softmax, gamma=-0.0003, eta=1.0),
    "clipped(-.0005:1)": partial(clipped_softmax, gamma=-0.0005, eta=1.0),
    "clipped(-.001:1)": partial(clipped_softmax, gamma=-0.001, eta=1.0),
    "clipped(-.002:1)": partial(clipped_softmax, gamma=-0.002, eta=1.0),
    "clipped(-.0025:1)": partial(clipped_softmax, gamma=-0.0025, eta=1.0),
    "clipped(-.003:1)": partial(clipped_softmax, gamma=-0.003, eta=1.0),
    "clipped(-.004:1)": partial(clipped_softmax, gamma=-0.004, eta=1.0),
    "clipped(-.005:1)": partial(clipped_softmax, gamma=-0.005, eta=1.0),
    "clipped(-.01:1)": partial(clipped_softmax, gamma=-0.01, eta=1.0),
    "clipped(-.015:1)": partial(clipped_softmax, gamma=-0.015, eta=1.0),
    "clipped(-.02:1)": partial(clipped_softmax, gamma=-0.02, eta=1.0),
    "clipped(-.025:1)": partial(clipped_softmax, gamma=-0.025, eta=1.0),
    "clipped(-.03:1)": partial(clipped_softmax, gamma=-0.03, eta=1.0),
    "clipped(-.04:1)": partial(clipped_softmax, gamma=-0.04, eta=1.0),
    "clipped(-.001:1.001)": partial(clipped_softmax, gamma=-0.001, eta=1.001),
    "clipped(-.002:1.002)": partial(clipped_softmax, gamma=-0.002, eta=1.002),
    "clipped(-.003:1.003)": partial(clipped_softmax, gamma=-0.003, eta=1.003),
    "clipped(-.005:1.005)": partial(clipped_softmax, gamma=-0.003, eta=1.005),
    "clipped(-.01:1.01)": partial(clipped_softmax, gamma=-0.01, eta=1.01),
    "clipped(-.03:1.03)": partial(clipped_softmax, gamma=-0.03, eta=1.03),
    "clipped(-.1:1.1)": partial(clipped_softmax, gamma=-0.1, eta=1.1),
}
