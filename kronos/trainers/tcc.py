"""TCC trainer.
"""

import torch

from kronos.trainers.base import Trainer
from kronos.losses.tcc import compute_tcc_loss


class TCCTrainer(Trainer):
    """A trainer for Temporal Cycle Consistency Learning [1].

    References:
        [1]: Temporal Cycle Consistency Learning,
        https://arxiv.org/abs/1904.07846
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

    def compute_loss(self, embs, steps, seq_lens, phase_labels=None):
        batch_size, num_cc_frames = embs.shape[:2]

        return compute_tcc_loss(
            embs,
            steps,
            seq_lens,
            stochastic_matching=self.stochastic_matching,
            normalize_embeddings=self.normalize_embeddings,
            loss_type=self.loss_type,
            similarity_type=self.similarity_type,
            num_cycles=int(batch_size * num_cc_frames),
            cycle_length=self.cycle_length,
            temperature=self.temperature,
            label_smoothing=self.label_smoothing,
            variance_lambda=self.variance_lambda,
            huber_delta=self.huber_delta,
            normalize_indices=self.normalize_indices,
        )
