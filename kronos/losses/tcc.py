"""Temporal cycle consistency loss.
"""

import torch
import torch.nn.functional as F

from kronos.losses import core


def compute_tcc_loss(
    embs,
    idxs,
    seq_lens,
    stochastic_matching=False,
    normalize_embeddings=False,
    loss_type="classification",
    similarity_type="l2",
    num_cycles=20,
    cycle_length=2,
    temperature=0.1,
    label_smoothing=0.1,
    variance_lambda=0.001,
    huber_delta=0.1,
    normalize_indices=True,
):
    """Computes TCC loss between sequences of embeddings.
    """
    msg = "Invalid similarity type."
    assert similarity_type in ["l2", "cosine"], msg
    msg = "Invalid loss type."
    assert loss_type in [
        "regression_mse_var",
        "regression_mse",
        "regression_huber",
        "classification",
    ], msg

    batch_size, num_cc = embs.shape[:2]

    if normalize_embeddings:
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)

    if stochastic_matching:
        return stochastic_tcc_loss(
            embs=embs,
            idxs=idxs,
            seq_lens=seq_lens,
            num_cc=num_cc,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
        )

    return deterministic_tcc_loss(
        embs=embs,
        idxs=idxs,
        seq_lens=seq_lens,
        num_cc=num_cc,
        batch_size=batch_size,
        loss_type=loss_type,
        similarity_type=similarity_type,
        temperature=temperature,
        label_smoothing=label_smoothing,
        variance_lambda=variance_lambda,
        huber_delta=huber_delta,
        normalize_indices=normalize_indices,
    )


def deterministic_tcc_loss(
    embs,
    idxs,
    seq_lens,
    num_cc,
    batch_size,
    loss_type,
    similarity_type,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
):
    """Deterministic alignment between all pairs of sequences in a batch."""

    batch_size = embs.shape[0]

    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                logits, labels = align_sequence_pair(
                    embs[i], embs[j], similarity_type, temperature
                )
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(idxs[i : i + 1].expand(num_cc, -1))
                seq_lens_list.append(seq_lens[i : i + 1].expand(num_cc))

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    if loss_type == "classification":
        return classification_loss(logits, labels, label_smoothing)
    return regression_loss(
        logits,
        labels,
        num_cc,
        steps,
        seq_lens,
        loss_type,
        normalize_indices,
        variance_lambda,
        huber_delta,
    )


def pairwise_l2_sq(x1, x2):
    """Compute pairwise squared Euclidean distances.
    """
    dot = torch.mm(x1.double(), torch.t(x2.double()))
    norm_sq = torch.diag(dot)
    dist = norm_sq[None, :] - 2 * dot + norm_sq[:, None]
    dist = torch.clamp(dist, min=0)  # replace negative values with 0
    return dist.float()


def pairwise_cosine(x1, x2, distance=True):
    """Compute pairwise cosine distances.
    """
    dist = torch.mm(x1.double(), torch.t(x2.double()))
    if distance:
        return (1 - dist).float()  # distance
    return dist.float()  # similarity


def softmax(x):
    """Compute row-wise softmax.

    Notes:
        Since the input to this softmax is the negative of the
        pairwise L2 distances, we don't need to do the classical
        numerical stability trick.
    """
    exp = torch.exp(x)
    return exp / exp.sum(dim=1, keepdim=True)


def get_scaled_similarity(emb1, emb2, similarity_type, temperature):
    """Return pairwise similarity.
    """
    embedding_dim = emb1.shape[1]
    if similarity_type == "l2":
        similarity = -1 * pairwise_l2_sq(emb1, emb2)
    else:
        similarity = pairwise_cosine(emb1, emb2, distance=False)
    similarity = similarity / embedding_dim
    similarity = similarity / temperature
    return similarity


def align_sequence_pair(emb1, emb2, similarity_type, temperature):
    """Align two sequences.
    """
    max_num_steps = emb1.shape[0]
    sim_12 = get_scaled_similarity(emb1, emb2, similarity_type, temperature)
    softmaxed_sim_12 = softmax(sim_12)
    nn_embs = torch.mm(softmaxed_sim_12, emb2)
    sim_21 = get_scaled_similarity(nn_embs, emb1, similarity_type, temperature)
    logits = sim_21
    labels = torch.arange(max_num_steps).to(logits.device)
    return logits, labels


def stochastic_tcc_loss(
    embs,
    idxs,
    seq_lens,
    num_cc,
    batch_size,
    loss_type,
    similarity_type,
    num_cycles,
    cycle_length,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
):
    """Stochastic alignment between randomly sampled cycles.
    """
    cycles = gen_cycles(num_cycles, batch_size, cycle_length)
    cycles = cycles.to(embs.device)

    logits, labels = align_find_cycles(
        cycles,
        embs,
        num_cc,
        num_cycles,
        cycle_length,
        similarity_type,
        temperature,
    )

    if loss_type == "classification":
        return classification_loss(logits, labels, label_smoothing)

    idxs = torch.index_select(idxs, 0, cycles[:, 0])
    seq_lens = torch.index_select(seq_lens, 0, cycles[:, 0])

    return regression_loss(
        logits,
        labels,
        num_cc,
        idxs,
        seq_lens,
        loss_type,
        normalize_indices,
        variance_lambda,
        huber_delta,
    )


def gen_cycles(num_cycles, batch_size, cycle_length):
    """Generates cycles for alignment.
    """
    idxs = torch.arange(batch_size).unsqueeze(0).repeat(num_cycles, 1)
    rand_idxs = torch.rand(num_cycles, batch_size).argsort(dim=1)
    cycles = torch.gather(idxs, 1, rand_idxs)
    cycles = cycles[:, :cycle_length]
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)
    return cycles


def align_find_cycles(
    cycles,
    embs,
    num_cc,
    num_cycles,
    cycle_length,
    similarity_type,
    temperature,
):
    logits_list = []
    labels_list = []
    for i in range(num_cycles):
        logits, labels = align_single_cycle(
            cycles[i],
            embs,
            cycle_length,
            num_cc,
            similarity_type,
            temperature,
        )
        logits_list.append(logits)
        labels_list.append(labels)
    logits = torch.stack(logits_list)
    labels = torch.cat(labels_list)
    return logits, labels


def align_single_cycle(
    cycle, embs, cycle_length, num_steps, similarity_type, temperature,
):
    """Take a single cycle and returns logits and labels.
    """
    # choose a random frame
    n_idx = torch.randint(num_steps, size=(1,))
    # create labels
    labels = n_idx.to(embs.device)

    # query the features of the randomly sampled frame
    # from the first video sequence in the cycle
    query_feats = embs[cycle[0], n_idx : n_idx + 1]

    num_channels = query_feats.shape[-1]
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]
        if similarity_type == "l2":
            mean_l2_sq = (
                (query_feats - candidate_feats).pow(2).sum(dim=-1).unsqueeze(1)
            )
            similarity = -mean_l2_sq
        else:  # cosine
            # dot product of embeddings
            similarity = torch.mm(candidate_feats, torch.t(query_feats))

        similarity = similarity / num_channels
        similarity = similarity / temperature

        if c == cycle_length:
            break

        # find weighted nn
        beta = F.softmax(similarity, dim=0)
        query_feats = (beta * candidate_feats).sum(dim=0, keepdim=True)

    similarity = similarity.squeeze()
    return similarity, labels


def classification_loss(logits, labels, label_smoothing):
    """Cycle-back classification loss.
    """
    return core.cross_entropy(logits, labels, label_smoothing)


def regression_loss(
    logits,
    labels,
    num_steps,
    steps,
    seq_lens,
    loss_type,
    normalize_indices,
    variance_lambda,
    huber_delta,
):
    """Cycle-back regression loss.
    """
    if normalize_indices:
        steps = steps.float() / seq_lens[:, None].float()
    else:
        steps = steps
    labels = core.one_hot(labels, logits.shape[1])
    beta = F.softmax(logits, dim=1)
    time_true = (steps * labels).sum(dim=1)
    time_pred = (steps * beta).sum(dim=1)
    if loss_type in ["regression_mse", "regression_mse_var"]:
        if "var" in loss_type:  # variance-aware regression
            # compute log of prediction variance
            time_pred_var = (steps - time_pred.unsqueeze(1)).pow(2) * beta
            time_pred_var = torch.log(time_pred_var.sum(dim=1))
            err_sq = (time_true - time_pred).pow(2)
            loss = (
                torch.exp(-time_pred_var) * err_sq
                + variance_lambda * time_pred_var
            )
            return loss.mean()
        return F.mse_loss(time_pred, time_true)
    return core.huber_loss(time_pred, time_true, huber_delta)
