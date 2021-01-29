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
            return 0.0
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

        if self.normalize_embeddings:
            embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)

        # Compute pairwise squared L2 distances between embeddings.
        embs_flat = embs.view(-1, num_dims)
        distances = torch.cdist(embs_flat, embs_flat)

        # Each row in a batch corresponds to a frame sequence. Since this
        # baseline assumes rough alignment between sequences, we want columns,
        # i.e. frames in each row that belong to the same index to be close
        # together in embedding space. Additionally, they should be apart from
        # every other frame in the entire batch.
        # This computes the mask accordingly.
        labels = torch.arange(num_cc_frames).unsqueeze(0).repeat(batch_size, 1)
        labels = labels.to(embs.device)
        mask = labels.flatten()[:, None] == labels.flatten()[None, :]

        # Compute positive and negative loss terms.
        pos_loss = (distances * mask.float()).sum(dim=-1).mean()
        # margin_diff = (1 - distances) * (~mask).float()
        # hinge_loss = torch.clamp(margin_diff, min=0).pow(2).sum(1).mean()

        # return pos_loss + hinge_loss

        return pos_loss
