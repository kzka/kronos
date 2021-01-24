import numpy as np
import torch

from scipy.spatial.distance import cdist
from torchvision.utils import make_grid

from kronos.evaluators.base import Evaluator
from kronos.datasets.transforms import UnNormalize


class NearestNeighbourVisualizer(Evaluator):
    """Nearest-neighbour frame visualizer.
    """

    def __init__(self, distance, num_frames, num_ctx_frames):
        """Constructor.

        Args:
            distance (str): The distance metric to use when
                calculating nearest-neighbours.
            num_frames (int): The number of frames to plot.
            num_ctx_frames (int): The number of context frames stacked
                together for each individual video frame.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        if distance not in ["sqeuclidean", "cosine"]:
            raise ValueError(
                "{} is not a supported distance metric.".format(distance)
            )

        self.distance = distance
        self.num_frames = num_frames
        self.num_ctx_frames = num_ctx_frames

    def _evaluate(self, embs, labels, frames, fit=False):
        """Get pairwise nearest-neighbour frames.
        """
        # reshape images
        for i in range(len(frames)):
            b, s, c, h, w = frames[i].shape
            seq_len = s // self.num_ctx_frames
            frames[i] = frames[i].view(
                b, seq_len, self.num_ctx_frames, c, h, w
            )
        # generate random query and candidate indices
        query_idx, cand_idx = np.random.choice(
            np.arange(len(embs)), size=2, replace=False
        )
        query_emb = embs[query_idx]
        cand_emb = embs[cand_idx]
        res, images = {}, []
        dists = cdist(query_emb, cand_emb, self.distance)
        # generate random query frames
        query_frame_idxs = np.random.choice(
            np.arange(len(query_emb)), size=self.num_frames, replace=False
        )
        query_frame_idxs.sort()
        for q_idx in query_frame_idxs:
            dist_query = dists[q_idx, :]
            frame_query = frames[query_idx][0:1, q_idx, -1]
            nn = np.argmin(dist_query)
            frame_cand = frames[cand_idx][0:1, nn, -1]
            img = torch.cat([frame_query, frame_cand], dim=0)
            img = make_grid(img, nrow=1)
            # images.append(UnNormalize()(img).permute(1, 2, 0))
            images.append(img.permute(1, 2, 0))
        res["image"] = torch.cat(images, dim=1).detach().cpu().numpy()
        return res
