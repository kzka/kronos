import numpy as np
import torch

from scipy.spatial.distance import cdist
from torchvision.utils import make_grid

from kronos.evaluators.base import Evaluator
from kronos.datasets.transforms import UnNormalize


class PhaseAlignmentTopK(Evaluator):
    """Phase alignment topk evaluation.

    Each action class contains a fixed number of phases.
    Given B videos, we align all `N x (N-1)` pairs and compute
    phase alignment accuracy for each phase in each pair.
    """

    def __init__(self, distance, topk, num_ctx_frames):
        """Constructor.

        Args:
            distance (str): The distance metric to use when
                calculating nearest-neighbours.
            topk (list): A list of integers, specifying the
                top-k values for which to compute the accuracy.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        if distance not in ["sqeuclidean", "cosine"]:
            raise ValueError(
                "{} is not a supported distance metric.".format(distance)
            )

        self.distance = distance
        self.topk = topk
        self.numk = len(topk)
        self.num_ctx_frames = num_ctx_frames

    def _evaluate(self, embs, labels, frames, fit=False, recons=None):
        """Get pairwise nearest-neighbours then calculate phase alignment accuracy.
        """
        # reshape images
        for i in range(len(frames)):
            b, s, c, h, w = frames[i].shape
            seq_len = s // self.num_ctx_frames
            frames[i] = frames[i].view(
                b, seq_len, self.num_ctx_frames, c, h, w
            )
        num_embs = len(embs)
        total_combinations = num_embs * (num_embs - 1)
        num_phases = len(labels[0].keys())
        accuracy = np.zeros((self.numk, total_combinations, num_phases))
        res, images, idx = {}, [], 0
        for i, (query_emb, query_label) in enumerate(zip(embs, labels)):
            for j, (cand_emb, cand_label) in enumerate(zip(embs, labels)):
                if i == j:
                    continue
                dists = cdist(query_emb, cand_emb, self.distance)
                for phase_idx, (ql, cl) in enumerate(
                    zip(query_label.values(), cand_label.values())
                ):
                    exists_query, t_query = ql
                    exists_cand, t_cand = cl
                    if exists_query and exists_cand:
                        dist_query = dists[t_query[0] : t_query[1] + 1, :]
                        if i == 0 and j == 1:
                            frames_query = frames[0][0:1, t_query[-1], -1]
                            nns = np.argmin(dist_query, axis=1)
                            frames_cand = frames[1][0:1, nns[-1], -1]
                            img = torch.cat([frames_query, frames_cand], dim=0)
                            img = make_grid(img, nrow=1)
                            images.append(UnNormalize()(img).permute(1, 2, 0))
                        for k_idx, k_val in enumerate(self.topk):
                            nns = np.argpartition(dist_query, k_val)[:, :k_val]
                            if any(
                                [
                                    n in np.arange(t_cand[0], t_cand[1] + 1)
                                    for n in nns.flatten()
                                ]
                            ):
                                accuracy[k_idx, idx, phase_idx] = 1
                idx += 1
        accuracy = np.mean(accuracy, axis=(1, 2))
        res["scalar"] = {
            "top-{}".format(k): v for k, v in zip(self.topk, accuracy)
        }
        res["image"] = torch.cat(images, dim=1).detach().cpu().numpy()
        return res


class PhaseAlignmentError(Evaluator):
    pass
