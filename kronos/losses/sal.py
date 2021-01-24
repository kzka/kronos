"""Shuffle and Learn loss.
"""

import numpy as np
import torch

from kronos.losses import core


def get_triplet_indices(
    batch_size, num_steps, num_samples, shuffle_fraction,
):
    """Generate positive and negative frame triplets.
    """
    total_num_samples = batch_size * num_samples

    # SAL paper terminology:
    # positive means not shuffled
    # negative means shuffled
    num_neg = int(shuffle_fraction * total_num_samples)
    num_pos = total_num_samples - num_neg

    labels = np.array([1] * num_neg + [0] * num_pos)
    np.random.shuffle(labels)

    # sample 5 frames (a, b, c, d, e) from a temporal window
    # such that a < b < c < d < e
    # terminology:
    # a - b - c - d - e - f
    # 0 - 1 - 2 - 3 - 4 - 5
    indices = np.random.rand(total_num_samples, num_steps).argsort(axis=-1)
    indices = np.sort(indices[:, :5], axis=1)

    # additional training examples are also created
    # by inverting the order of all training instances
    reverse = np.random.random(total_num_samples) < 0.5
    indices[reverse] = indices[reverse][:, ::-1]

    # positive instances are created using (b, c, d)
    # i.e. (1, 2, 3).
    # negative instances are created using (b, a, d)
    # and (b, e, d), i.e. (1, 0, 3) and (1, 4, 3).
    # it is critical to use the same beginning frame
    # b and ending frame d while only changing the
    # middle frame for both positive and negative
    # examples.
    neg_samples = np.where(
        np.random.random((total_num_samples, 1)) < 0.5,
        np.take(indices, [1, 0, 3], axis=1),
        np.take(indices, [1, 4, 3], axis=1),
    )
    pos_samples = np.take(indices, [1, 2, 3], axis=1)
    indices = np.where(labels[:, None], neg_samples, pos_samples)

    # convert to torch tensors
    indices = torch.from_numpy(indices).long()
    labels = torch.from_numpy(labels).long()

    return indices, labels


def sample_triplets(embs, shuffle_fraction, num_samples):
    """Samples shuffled triplets across a mini-batch.
    """
    batch_size, num_steps = embs.shape[:2]
    indices, labels = get_triplet_indices(
        batch_size, num_steps, num_samples, shuffle_fraction,
    )
    indices = indices.to(embs.device)
    labels = labels.to(embs.device)
    embs = embs.repeat(num_samples, 1, 1)
    embs = torch.gather(
        embs, 1, indices.unsqueeze(-1).repeat(1, 1, embs.shape[-1]),
    )
    embs = embs.view(embs.shape[0], -1)
    return embs, labels


def compute_sal_loss(
    embs, classifier, shuffle_fraction, num_samples, label_smoothing,
):
    """Computes SAL loss between sequences of embeddings.

    Args:
        embs (torch.FloatTensor): The raw embeddings output by the
            TCC network of shape (B, T, D) where B is the batch size,
            T is the number of sampled frames in the sequence and D
            is the dimension of the embedding space.
        classifier (nn.Module): The SAL classifier.
        shuffle_fraction (float): The percentage of shuffled frames
            across the whole mini-batch of videos.
        num_samples (int): The number of triplets (positive and negative)
            to sample in one video sequence.
    """
    assert (
        0 <= shuffle_fraction <= 1
    ), "Shuffle fraction must be between 0 and 1."
    assert embs.ndim == 3, "Embeddings must be of shape (B, T, D)."
    embs_sal, labels = sample_triplets(embs, shuffle_fraction, num_samples)
    logits = classifier(embs_sal)
    return core.cross_entropy(logits, labels, label_smoothing)
