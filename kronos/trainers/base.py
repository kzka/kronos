"""Base class for defining training algorithms.
"""

import abc

import logging
import torch

try:
    from apex import amp

    IS_APEX = True
except ImportError:
    logging.info(
        "apex package is not installed. Mixed precision training disabled."
    )
    IS_APEX = False


class Trainer(abc.ABC):
    """Base trainer abstraction.

    Subclasses should override `compute_loss`.
    """

    def __init__(
        self, model, optimizer, device, opt_lvl=0, config=None,
    ):
        """Constructor.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim): The optimizer to use.
            device (torch.device): The computing device.
            opt_level (int): An int specifying the optimization level
                for mixed precision.
            config (dict): Extra sampling or model related configuration
                variables.
        """
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._opt_lvl = opt_lvl
        self._config = config

        self._model.train()
        self._model.to(self._device)

        # mixed precision training
        if self._opt_lvl > 0 and IS_APEX:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=f"O{self._opt_lvl}"
            )

    @abc.abstractmethod
    def compute_loss(self, embs, steps, seq_lens, phase_labels=None):
        """Compute the loss on a single batch.

        Args:
            embs (torch.FloatTensor): The output of the embedding
                network.
            steps (torch.LongTensor): The frame indices for the
                frames of each video in the batch.
            seq_lens (torch.LongTensor) The original length or
                number of frames of each video in the batch.

        Returns:
            A tensor corresponding to the value of the loss
            function evaluated on the given batch.

        :meta public:
        """
        pass

    def train_one_iter(self, batch, global_step):
        """Single forward + backward pass of the model.

        Args:
            batch (dict): The output of a dataloader wrapping
                a `kronos.datasets.VideoDataset`.
            global_step (int): The training iteration number.
        """
        self._model.train()

        frames = batch["frames"].to(self._device)
        steps = batch["frame_idxs"].to(self._device)
        seq_lens = batch["video_len"].to(self._device)
        phase_labels = None
        if "phase_labels" in batch:
            phase_labels = batch["phase_labels"].to(self._device)

        # clear optimizer cache
        self._optimizer.zero_grad()

        # forward through model
        embs = self._model(frames)["embs"]

        # compute loss
        loss = self.compute_loss(embs, steps, seq_lens, phase_labels)

        # scale loss for mixed precision
        if self._opt_lvl > 0 and IS_APEX:
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # gradient step
        self._optimizer.step()

        return loss

    @torch.no_grad()
    def eval_num_iters(self, valid_loader, eval_iters=None):
        """Evaluate the model on the validation data.

        Args:
            valid_loader: The validation data loader.
            eval_iters (int): The number of time to call
                `next` on the data iterator. Set to `None`
                to evaluate on the whole validation set.
        """
        self._model.eval()

        val_loss = 0.0
        it_ = 0
        for batch_idx, batch in enumerate(valid_loader):
            if eval_iters is not None and batch_idx >= eval_iters:
                break

            frames = batch["frames"].to(self._device)
            steps = batch["frame_idxs"].to(self._device)
            seq_lens = batch["video_len"].to(self._device)
            phase_labels = None
            if "phase_labels" in batch:
                phase_labels = batch["phase_labels"].to(self._device)

            # forward through model
            embs = self._model(frames)["embs"]

            # compute loss
            val_loss += self.compute_loss(embs, steps, seq_lens, phase_labels)
            it_ += 1

        val_loss /= it_
        return val_loss
