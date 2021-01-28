import numpy as np

from itertools import permutations
from scipy.spatial.distance import cdist

from kronos.evaluators.base import Evaluator


class CycleConsistency(Evaluator):
    """2 and 3-way cycle consistency evaluator [1].

    References:
        [1]: Playing hard exploration games by watching YouTube,
        https://arxiv.org/abs/1805.11592
    """

    def __init__(self, mode, stride, distance):
        """Constructor.

        Args:
            mode (str): Specifies whether to use two or three-way
                cycle consistency. Can be one of `['two-way',
                'three-way', 'both']`.
            stride (int): Controls how many frames are skipped
                in each video sequence. For example, if the
                embedding vector of the first video is (100, 128),
                a stride of 5 reduces it to (20, 128).
            distance (str): The distance metric to use when
                calculating nearest-neighbours.

        Raises:
            ValueError: If the distance metric is invalid or the
                mode is invalid.
        """
        if distance not in ["sqeuclidean", "cosine"]:
            raise ValueError(
                "{} is not a supported distance metric.".format(distance)
            )
        if mode not in ["two_way", "three_way", "both"]:
            raise ValueError("{} is not a supported mode.".format(mode))

        self.stride = int(stride)
        self.distance = distance
        self.mode = mode

    def _evaluate_two_way(self, embs):
        """Two-way cycle consistency.
        """
        num_embs = len(embs)
        total_combinations = num_embs * (num_embs - 1)
        ccs = np.zeros((total_combinations))
        idx, res = 0, {}
        for i in range(num_embs):
            query_emb = embs[i][:: self.stride]
            ground_truth = np.arange(len(embs[i]))[:: self.stride]
            for j in range(num_embs):
                if i == j:
                    continue
                candidate_emb = embs[j][:: self.stride]
                dists = cdist(query_emb, candidate_emb, self.distance)
                nns = np.argmin(dists[:, np.argmin(dists, axis=1)], axis=0)
                ccs[idx] = np.mean(np.abs(nns - ground_truth) <= 1)
                idx += 1
        ccs = ccs[~np.isnan(ccs)]
        res["scalar"] = np.mean(ccs)
        return res

    def _evaluate_three_way(self, embs):
        num_embs = len(embs)
        cycles = np.stack(list(permutations(np.arange(num_embs), 3)))
        total_combinations = len(cycles)
        ccs = np.zeros((total_combinations))
        res = {}
        for c_idx, cycle in enumerate(cycles):
            # forward consistency check.
            # each cycle will be a length 3 permutation,
            # e.g. U - V - W.
            # we compute nearest neighbours across consecutive
            # pairs in the cycle and loop back to the first
            # cycle index to obtain:
            # U - V - W - U.
            query_emb = None
            for i in range(len(cycle)):
                if query_emb is None:
                    query_emb = embs[cycle[i]][:: self.stride]
                candidate_emb = embs[cycle[(i + 1) % len(cycle)]][
                    :: self.stride
                ]
                dists = cdist(query_emb, candidate_emb, self.distance)
                nns_forward = np.argmin(dists, axis=1)
                query_emb = candidate_emb[nns_forward]
            ground_truth_forward = np.arange(len(embs[cycle[0]]))[
                :: self.stride
            ]
            cc_forward = np.abs(nns_forward - ground_truth_forward) <= 1
            # backward consistency check.
            # a backward check is equivalent to
            # reversing the middle pair V - W
            # and performing a forward check,
            # e.g. U - W - V - U.
            cycle[1:] = cycle[1:][::-1]
            query_emb = None
            for i in range(len(cycle)):
                if query_emb is None:
                    query_emb = embs[cycle[i]][:: self.stride]
                candidate_emb = embs[cycle[(i + 1) % len(cycle)]][
                    :: self.stride
                ]
                dists = cdist(query_emb, candidate_emb, self.distance)
                nns_backward = np.argmin(dists, axis=1)
                query_emb = candidate_emb[nns_backward]
            ground_truth_backward = np.arange(len(embs[cycle[0]]))[
                :: self.stride
            ]
            cc_backward = np.abs(nns_backward - ground_truth_backward) <= 1
            # require consistency both ways
            cc = np.logical_and(cc_forward, cc_backward)
            ccs[c_idx] = np.mean(cc)
        ccs = ccs[~np.isnan(ccs)]
        res["scalar"] = np.mean(ccs)
        return res

    def _evaluate(self, embs, labels, frames, fit=False, recons=None):
        if self.mode == "two_way":
            return self._evaluate_two_way(embs)
        elif self.mode == "three_way":
            return self._evaluate_three_way(embs)
        else:
            res_two_way = self._evaluate_two_way(embs)
            res_three_way = self._evaluate_three_way(embs)
            return {
                "scalar": {
                    "two_way": res_two_way["scalar"],
                    "three_way": res_three_way["scalar"],
                }
            }
