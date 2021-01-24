"""TCC trainer.
"""

import torch

from kronos.trainers.base import Trainer
from kronos.losses.tcc import compute_tcc_loss
from ipdb import set_trace


class SubTCCTrainer(Trainer):
    """Sub-trajectory TCC alignment trainer.
    """

    def __init__(self, model, optimizer, device, opt_lvl, config):
        super().__init__(model, optimizer, device, opt_lvl, config)

        self.stochastic_matching = config.LOSS.STOCHASTIC_MATCHING
        self.normalize_embeddings = config.LOSS.L2_NORMALIZE_EMBEDDINGS
        self.loss_type = config.LOSS.LOSS_TYPE
        self.similarity_type = config.LOSS.SIMILARITY_TYPE
        self.cycle_length = config.LOSS.CYCLE_LENGTH
        self.temperature = config.LOSS.SOFTMAX_TEMPERATURE
        self.label_smoothing = config.LOSS.LABEL_SMOOTHING
        self.variance_lambda = config.LOSS.VARIANCE_LAMBDA
        self.huber_delta = config.LOSS.HUBER_DELTA
        self.normalize_indices = config.LOSS.NORMALIZE_INDICES

        self.weight_tcc = config.LOSS.WEIGHT_TCC
        self.weight_hinge = 1 - config.LOSS.WEIGHT_TCC
        self.margin_hinge = config.LOSS.MARGIN_HINGE

    @staticmethod
    def _pairwise_l2_sq(x):
        """Compute pairwise squared Euclidean distances.
        """
        dot = torch.mm(x.double(), torch.t(x.double()))
        norm_sq = torch.diag(dot)
        dist = norm_sq[None, :] - 2 * dot + norm_sq[:, None]
        dist = torch.clamp(dist, min=0)  # replace negative values with 0
        return dist.float()

    def compute_loss(self, embs, steps, seq_lens, phase_labels):
        batch_size, num_cc_frames, num_dims = embs.shape

        # loop through the phases and compute tcc
        # loss on frames with the same phase label
        tcc_loss = []
        num_phases = torch.unique(phase_labels)
        for phase_idx in num_phases:
            same_phase_mask = phase_labels.eq(phase_idx)
            row_sum = same_phase_mask.sum(dim=1)
            non_zero_min = row_sum[row_sum > 0]
            min_frames = non_zero_min.min()
            for i in range(batch_size):
                num_true = 0
                for j in range(same_phase_mask.shape[1]):
                    if same_phase_mask[i, j]:
                        num_true += 1
                    if num_true > min_frames:
                        same_phase_mask[i, j] = False
            same_phase_batch_size = len(same_phase_mask.sum(dim=1).nonzero())
            if same_phase_batch_size < 2:
                continue
            same_phase_embs = embs[same_phase_mask].view(
                same_phase_batch_size, -1, num_dims
            )
            same_phase_steps = steps[same_phase_mask].view(
                same_phase_batch_size, -1
            )
            same_phase_seq_lens = seq_lens[
                same_phase_mask.sum(dim=1).nonzero()
            ].squeeze()
            num_cycles = int(same_phase_batch_size * same_phase_embs.shape[1])
            tcc_loss.append(
                compute_tcc_loss(
                    same_phase_embs,
                    same_phase_steps,
                    same_phase_seq_lens,
                    stochastic_matching=self.stochastic_matching,
                    normalize_embeddings=self.normalize_embeddings,
                    loss_type=self.loss_type,
                    similarity_type=self.similarity_type,
                    num_cycles=num_cycles,
                    cycle_length=self.cycle_length,
                    temperature=self.temperature,
                    label_smoothing=self.label_smoothing,
                    variance_lambda=self.variance_lambda,
                    huber_delta=self.huber_delta,
                    normalize_indices=self.normalize_indices,
                )
            )
        tcc_loss = torch.stack(tcc_loss).mean()

        # now loop and compute squared hinge loss on frames with
        # different phase labels
        hinge_loss = []
        for phase_idx in num_phases:
            diff_phase_mask = ~phase_labels.eq(phase_idx).flatten()
            embs_flat = embs.view(-1, num_dims)
            distances = torch.cdist(embs_flat, embs_flat)
            margin_dist = self.margin_hinge - distances
            margin_diff = margin_dist * diff_phase_mask.float()
            hinge_loss.append(
                torch.clamp(margin_diff, min=0).pow(2).sum(1).mean()
            )
        hinge_loss = torch.stack(hinge_loss).mean()

        return (self.weight_tcc * tcc_loss) + (self.weight_hinge * hinge_loss)
