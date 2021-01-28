import numpy as np
import torch
import torch.nn.functional as F

from torchvision.utils import make_grid

from kronos.evaluators.base import Evaluator


class ReconstructionVisualizer(Evaluator):
    """Reconstructed frame visualization."""

    def __init__(self, num_frames, num_ctx_frames):
        """Constructor.

        Args:
            num_frames (int): The number of frames to plot.
            num_ctx_frames (int): The number of context frames stacked
                together for each individual video frame.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        self.num_frames = num_frames
        self.num_ctx_frames = num_ctx_frames

    def _evaluate(self, embs, labels, frames, fit=False, recons=None):
        del labels
        del embs

        # Reshape images.
        for i in range(len(frames)):
            b, s, c, h, w = frames[i].shape
            seq_len = s // self.num_ctx_frames
            frames[i] = frames[i].view(b, seq_len, self.num_ctx_frames, c, h, w)

        r_idx = np.random.randint(0, len(frames))
        frame = frames[r_idx].squeeze()
        recon = recons[r_idx]

        # Downsample the frame.
        _, _, sh, sw = recon.shape
        _, _, h, w = frame.shape
        scale_factor = sh / h
        frame_ds = F.interpolate(
            frame,
            mode='bilinear',
            scale_factor=scale_factor,
            recompute_scale_factor=False,
            align_corners=True)

        frame_idxs = np.random.choice(
            np.arange(frame_ds.shape[0]), size=self.num_frames, replace=False)

        # Subsample num_frames to plot.
        frame_ds = frame_ds[frame_idxs]
        recon = recon[frame_idxs]

        # Clip reconstructin between 0 and 1.
        recon = torch.clamp(recon, 0.0, 1.0)

        imgs = torch.cat([frame_ds, recon], axis=0)
        img = make_grid(imgs, nrow=2)

        return {
            "image": img.permute(1, 2, 0).detach().cpu().numpy()
        }
