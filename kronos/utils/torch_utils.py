"""Pytorch-related utils.
"""

import torch


def freeze_model(model, bn_freeze_affine=False, bn_use_running_stats=False):
    """Freeze model weights.

    Args:
        bn_freeze_affine (bool): Set to `True` to freeze batch norm
            params gamma and beta.
        bn_use_running_stats (bool): Set to `True` to switch from batch
            statistics to running mean and std. This is recommended
            for very small batch sizes.
    """
    for m in model.modules():
        if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad = False
            m.eval()
        else:
            if bn_freeze_affine:
                for p in m.parameters(recurse=False):
                    p.requires_grad = False
            else:
                for p in m.parameters(recurse=False):
                    p.requires_grad = True
            if bn_use_running_stats:
                m.eval()
