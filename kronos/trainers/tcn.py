"""TCN trainers.
"""

from kronos.trainers.base import Trainer
from kronos.losses import tcn


class SVTCNTrainer(Trainer):
    """A trainer for Single-View TCN [1].

    References:
        [1]: Time-Contrastive Networks,
        https://arxiv.org/abs/1704.06888
    """

    def __init__(self, model, optimizer, device, opt_lvl, config):
        super().__init__(model, optimizer, device, opt_lvl, config)

        self.normalize_embeddings = config.LOSS.L2_NORMALIZE_EMBEDDINGS
        self.loss_type = config.LOSS.LOSS_TYPE
        self.pos_radius = config.LOSS.POS_RADIUS
        self.neg_radius = config.LOSS.NEG_RADIUS
        self.margin = config.LOSS.MARGIN

    def compute_loss(self, embs, steps, seq_lens):
        return tcn.compute_svtcn_loss(
            embs,
            steps,
            seq_lens,
            pos_radius=self.pos_radius,
            neg_radius=self.neg_radius,
            margin=self.margin,
            normalize_embeddings=self.normalize_embeddings,
            loss_type=self.loss_type,
        )


class MVTCNTrainer(Trainer):
    """A trainer for Multi-View TCN [1].

    References:
        [1]: Time-Contrastive Networks,
        https://arxiv.org/abs/1704.06888
    """

    pass
