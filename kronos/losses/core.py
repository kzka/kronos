"""Useful loss methods.
"""

import torch
import torch.nn.functional as F


def one_hot(y, K, smooth_eps=0):
    """1-hot encodes a tensor with optional smoothing.

    Args:
        y (tensor): A tensor containing the ground
            truth labels of shape (N,), i.e. one label
            for each element in the batch.
        K (int): The number of classes.
        smooth_eps (float): Label smoothing factor.
    """
    assert 0 <= smooth_eps <= 1
    y_hot = torch.eye(K)[y] * (1 - smooth_eps) + (smooth_eps / (K - 1))
    y_hot = y_hot.to(y.device)
    return y_hot


def cross_entropy(logits, labels, smooth_eps=0, reduction="mean"):
    """Cross-entropy loss with support for label smoothing.

    Args:
        logits (tensor): A FloatTensor containing the raw logits,
            i.e. no softmax has been applied to the model output.
            The tensor should be of shape (N, K) where K is the
            number of classes.
        labels (tensor): A LongTensor containing the ground truth
            labels, of shape (N,).
        smooth_eps (float): The label smoothing factor. This should
            be a float between 0 and 1.
        reduction (str): The reduction strategy on the final loss
            value.
    """
    assert isinstance(logits, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(labels, (torch.LongTensor, torch.cuda.LongTensor))

    # ensure logits are not 1-hot encoded
    assert labels.ndim == 1, "[!] Labels are not expected to be 1-hot encoded."

    if smooth_eps == 0:
        return F.cross_entropy(logits, labels, reduction=reduction)

    # one hot encode targets
    num_classes = logits.shape[1]
    labels = one_hot(labels, num_classes, smooth_eps)

    # convert logits to log probas
    log_probs = F.log_softmax(logits, dim=-1)

    loss = (-labels * log_probs).sum(dim=-1)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum(dim=-1)
    else:
        raise ValueError("Unsupported reduction method.")


def huber_loss(input, target, delta):
    """Huber loss.
    """
    diff = target - input
    diff_abs = torch.abs(target - input)
    cond = diff_abs <= delta
    loss = torch.where(
        cond, 0.5 * diff ** 2, delta * diff_abs - (0.5 * delta ** 2)
    )
    return loss.mean()
