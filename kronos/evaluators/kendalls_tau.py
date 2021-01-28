import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from kronos.evaluators.base import Evaluator


class KendallsTau(Evaluator):
    """Kendall rank correlation coefficient [1].

    References:
        [1]: Wikipedia article,
        https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    """

    def __init__(self, stride, distance):
        """Constructor.

        Args:
            stride (int): Controls how many frames are skipped
                in each video sequence. For example, if the
                embedding vector of the first video is (100, 128),
                a stride of 5 reduces it to (20, 128).
            distance (str): The distance metric to use when
                calculating nearest-neighbours.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        if distance not in ["sqeuclidean", "cosine"]:
            raise ValueError(
                "{} is not a supported distance metric.".format(distance)
            )

        self.stride = int(stride)
        self.distance = distance

    def _softmax(self, dists, temp=1.0):
        exp = np.exp(np.array(dists) / temp)
        dist = exp / np.sum(exp)
        return dist

    def _evaluate(self, embs, labels, frames, fit=False, recons=None):
        """Get pairwise nearest-neighbours then calculate Kendall's Tau.
        """
        num_embs = len(embs)
        total_combinations = num_embs * (num_embs - 1)
        taus = np.zeros((total_combinations))
        idx = 0
        res = {}
        for i in range(num_embs):
            query_emb = embs[i][:: self.stride]
            for j in range(num_embs):
                if i == j:
                    continue
                candidate_emb = embs[j][:: self.stride]
                dists = cdist(query_emb, candidate_emb, self.distance)
                if i == 0 and j == 1:
                    sim_matrix = []
                    for k in range(len(query_emb)):
                        sim_matrix.append(self._softmax(-dists[k]))
                    res["image"] = np.array(sim_matrix, dtype=np.float32)[
                        ..., None
                    ]
                nns = np.argmin(dists, axis=1)
                taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
                idx += 1
        taus = taus[~np.isnan(taus)]
        res["scalar"] = np.mean(taus)
        return res
