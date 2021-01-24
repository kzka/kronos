"""Triplet and n-pairs loss.
"""

import torch
import torch.nn.functional as F


def pairwise_l2_sq(x):
    """Compute pairwise squared Euclidean distances.
    """
    dot = torch.mm(x.double(), torch.t(x.double()))
    norm_sq = torch.diag(dot)
    dist = norm_sq[None, :] - 2 * dot + norm_sq[:, None]
    dist = torch.clamp(dist, min=0)  # replace negative values with 0
    return dist.float()


def compute_svtcn_loss(
    embs,
    idxs,
    seq_lens,
    pos_radius,
    neg_radius,
    margin=1.0,
    normalize_embeddings=True,
    loss_type="triplet_semihard",
):
    """Single-view TCN loss.
    """
    # assert neg_radius > pos_radius
    # assert (
    #     loss_type == "triplet_semihard"
    # ), "Only triplet loss is currently supported."

    # batch_size, num_frames = embs.shape[:2]
    # device = embs.device
    # idxs = idxs.flatten()

    # if normalize_embeddings:
    #     embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)

    # # reshape to (N, D)
    # embs = embs.view(batch_size * num_frames, -1)

    # # compute pairwise distances
    # pairwise_dists = pairwise_l2_sq(embs)

    # # create a sequence adjacency matrix,
    # # where `seq_adj[i, j] = True` if i
    # # and j belong to the same sequence
    # seq_ids = torch.cat(
    #     [torch.ones(num_frames) * i for i in range(batch_size)]
    # ).to(device)
    # seq_adj = seq_ids[:, None] == seq_ids[None, :]

    # # invert it so we also have a mask
    # # for selecting embeddings that do
    # # not belong to the same sequence
    # # and as such can also be used for
    # # negatives
    # seq_adj_not = ~seq_adj

    # # figure out which indices are within the positive
    # # range.
    # # for index j to be in the positive range of i,
    # # it must meet two conditions:
    # # 1) | i - j | <= pos_radius
    # # 2) i and j belong to the same seq, i.e. seq_adj[i, j] = True
    # pos_mask = torch.logical_and(
    #     torch.abs(idxs[:, None] - idxs[None, :]) <= pos_radius, seq_adj,
    # ).float()
    # # ensure `pos[i, i] = False`
    # pos_mask.fill_diagonal_(0)

    # # figure out which indices are within the negative
    # # range.
    # # for index j to be in the negative range if i,
    # # it must meet one of two conditions:
    # # 1) | i - j | > neg_radius
    # # 2) i and j don't belong to the same seq, i.e. seq_adj[i, j] = False
    # neg_mask = torch.logical_or(
    #     torch.abs(idxs[:, None] - idxs[None, :]) > neg_radius, seq_adj_not,
    # ).float()

    raise NotImplementedError
