"""Contrastive trainer."""

import torch
import torch.nn.functional as F

from kronos.trainers.base import Trainer


class ContrastiveTrainer(Trainer):
    """A trainer for contrastive learning."""

    def __init__(self, model, optimizer, device, opt_lvl, config):
        super().__init__(model, optimizer, device, opt_lvl, config)

        self.normalize_embeddings = config.LOSS.L2_NORMALIZE_EMBEDDINGS
        self.aux_loss = config.LOSS.AUX_LOSS.TYPE

    def compute_aux_loss(self, frames, reconstruction, steps, seq_lens):
        if self.aux_loss == 'none':
            return None
        elif self.aux_loss == 'autoencoding':
            _, _, sh, sw = reconstruction.shape
            _, _, c, h, w = frames.shape
            scale_factor = sh / h
            frames = frames.view(-1, c, h, w)
            frames_ds = F.interpolate(
                frames,
                mode='bilinear',
                scale_factor=scale_factor,
                recompute_scale_factor=False,
                align_corners=True)
            return F.mse_loss(reconstruction, frames_ds)

    def compute_loss(self, embs, steps, seq_lens, phase_labels=None):
        batch_size, num_cc_frames, num_dims = embs.shape

        # Compute pairwise L2 distances between embeddings.
        embs_flat = embs.view(-1, num_dims)
        distances = torch.cdist(embs_flat, embs_flat)

        # Zero out distances that belong to the same sequence.
        labels = torch.arange(batch_size).unsqueeze(1).repeat(1, num_cc_frames)
        labels = labels.to(embs.device)
        mask = labels.flatten()[:, None] == labels.flatten()[None, :]
        distances = distances * (~mask).float()

        return distances.mean()
