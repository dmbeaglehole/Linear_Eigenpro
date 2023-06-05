import os
import sys
import torch
from einops import rearrange

import numpy as np
import jax.numpy as jnp
from math import sqrt


def check_nan(A):
    assert(torch.sum(torch.isnan(A))==0)
    assert(torch.sum(torch.isinf(A))==0)
    return

def to_double(params):
    new_params = []
    for p in params:
        new_p = []
        for x in p:
            new_p.append(x.astype('float64'))
        new_params.append(tuple(new_p))
    return tuple(new_params)
