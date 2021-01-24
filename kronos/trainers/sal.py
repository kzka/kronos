"""SAL trainer.
"""

from kronos.trainers.base import Trainer
from kronos.losses.sal import compute_sal_loss


class SALTrainer(Trainer):
    """A trainer for Shuffle-and-Learn [1].

    References:
        [1]: Shuffle and Learn: Unsupervised Learning using Temporal
        Order Verification, https://arxiv.org/abs/1603.08561
    """

    def __init__(
        self, model, optimizer, device, opt_lvl, config,
    ):
        super().__init__(model, optimizer, device, opt_lvl, config)

        self.shuffle_fraction = config.SAMPLING.SHUFFLE_FRACTION
        self.num_samples = config.SAMPLING.NUM_SAMPLES
        self.label_smoothing = config.LOSS.LABEL_SMOOTHING

    def compute_loss(self, embs, steps, seq_lens):
        return compute_sal_loss(
            embs,
            classifier=self._model.auxiliary_net,
            shuffle_fraction=self.shuffle_fraction,
            num_samples=self.num_samples,
            label_smoothing=self.label_smoothing,
        )
