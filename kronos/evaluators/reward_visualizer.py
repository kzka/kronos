import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb

from kronos.evaluators.base import Evaluator


class RewardVisualizer(Evaluator):
    """Distance to goal state visualizer."""

    def __init__(self, distance, l2_normalize, num_plots=2):
        """Constructor.

        Args:
            distance (str): The distance metric to use when
                calculating nearest-neighbours.
            l2_normalize (bool): Whether to l2 normalize embeddings before
                computing distances.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        if distance not in ["sqeuclidean", "cosine"]:
            raise ValueError(
                "{} is not a supported distance metric.".format(distance)
            )

        self.distance = distance
        self.l2_normalize = l2_normalize
        self.num_plots = num_plots

    def _gen_reward_plot(self, rewards):
        """Create a pyplot plot and save to buffer."""
        fig, axes = plt.subplots(1, len(rewards),
            figsize=(6.4*self.num_plots, 4.8))
        for i, rew in enumerate(rewards):
            axes[i].plot(rew)
        fig.text(0.5, 0.04, 'Timestep', ha='center')
        fig.text(0.04, 0.5, 'Reward', va='center', rotation='vertical')
        fig.canvas.draw()
        img_arr = np.array(fig.canvas.renderer.buffer_rgba())
        return img_arr[:, :, :3]

    def _compute_goal_emb(self, embs):
        goal_emb = [emb[-1, :] for emb in embs]
        goal_emb = np.stack(goal_emb, axis=0)
        if self.l2_normalize:
            goal_emb /= np.linalg.norm(goal_emb, axis=-1, keepdims=True)
        goal_emb = np.mean(goal_emb, axis=0, keepdims=True)
        return goal_emb

    def _evaluate(self, embs, labels, frames, fit=False):
        goal_emb = self._compute_goal_emb(embs)
        rand_idxs = np.random.choice(
            np.arange(len(embs)), size=self.num_plots, replace=False)
        rewards = []
        for idx in rand_idxs:
            emb = embs[idx]
            if self.l2_normalize:
                emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
            dists = np.linalg.norm(emb - goal_emb, axis=-1)
            rewards.append(-dists)
        return {
            "image": self._gen_reward_plot(rewards),
        }
