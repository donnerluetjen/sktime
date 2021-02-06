#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Ansgar Asseburg"]
__all__ = [
    "UNIVARIATES",
    "MULTIVARIATES",
    "SAMPLE"
]

import pandas as pd
from sktime.utils._testing.series import _make_series
from numpy import random as rd
import numpy as np

RANDOM_SEED = 42

UNIVARIATES = [
    [rd.uniform(50, 100, (1, rd.randint(50, 100))),
     rd.uniform(50, 100, (1, rd.randint(50, 100)))],
    [rd.uniform(-50, 100, (1, rd.randint(50, 100))),
     rd.uniform(-50, 100, (1, rd.randint(50, 100)))]
]

MULTIVARIATES = [
    [rd.uniform(50, 100, (rd.randint(2, 10), rd.randint(50, 100))),
     rd.uniform(50, 100, (rd.randint(2, 10), rd.randint(50, 100)))],
    [rd.uniform(-50, 100, (rd.randint(2, 10), rd.randint(50, 100))),
     rd.uniform(-50, 100, (rd.randint(2, 10), rd.randint(50, 100)))]
]

AGDTW_SAMPLE = [
    [np.array([[5, 7, 4, 4, 3, 2]]),
     np.array([[1, 2, 3, 2, 2]]),
     3.73575899489195e+00 / 7],
    [np.array([[1, 2, 3, 4, 5]]),
     np.array([[1, 2, 3, 4, 5]]),
     5.0 / 5]
]

KERNEL_TEST_SAMPLE = [
    [np.array([5, 7, 4, 4, 3, 2]),
     np.array([1, 2, 3, 2, 2]),
     {'similarity': 3.73575899489195e+00, 'wp_length': 7}],
    [np.array([1, 2, 3, 4, 5]),
     np.array([1, 2, 3, 4, 5]),
     {'similarity': 5.0, 'wp_length': 5}]
]

NAN_SAMPLES = [
    [np.concatenate((rd.uniform(50, 100, (1, rd.randint(50, 100))),
                     np.array([[np.NaN]]),
                     rd.uniform(50, 100, (1, rd.randint(50, 100)))),
                    axis=1),
     rd.uniform(50, 100, (1, rd.randint(50, 100)))],
    [rd.uniform(-50, 100, (1, rd.randint(50, 100))),
     np.concatenate((rd.uniform(50, 100, (1, rd.randint(50, 100))),
                     np.array([[np.NaN]]),
                     rd.uniform(50, 100, (1, rd.randint(50, 100)))),
                    axis=1)]
]
