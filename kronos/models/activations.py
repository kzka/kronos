import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x, inplace=False):
    if inplace:
        # note the use of `sigmoid` as opposed
        # to `sigmoid_`.
        # this is because `x` is required
        # to compute the gradient during
        # the backward pass and calling
        # sigmoid in place will modify
        # its values.
        return x.mul_(x.sigmoid())
    return x * x.sigmoid()


class Swish(nn.Module):
    """Swish activation function [1].

    References:
        [1]: Searching for Activation Functions,
        https://arxiv.org/abs/1710.05941
    """

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)
