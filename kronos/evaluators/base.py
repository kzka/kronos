import abc

import logging
import math
import numpy as np
import torch

from tqdm import tqdm

from kronos.utils.py_utils import dict_mean


class Evaluator(abc.ABC):
    """Base abstraction for evaluating a temporal network on downstream tasks.

    Subclasses must implement the `_evaluate` method.
    """

    @abc.abstractmethod
    def _evaluate(self, embs, labels=None, frames=None, fit=False):
        """Evaluate the downstream task in embedding space.

        Args:
            embs (list): A list containing an embedding vector
                of every frame for every video in the validation
                dataset.
            labels (list): A list containing dense or sparse
                frame labels.
            frames (list) A list of videos stored as frame
                sequences we'd potentially like to plot.

        :meta public:
        """
        pass

    @torch.no_grad()
    def evaluate(
        self,
        global_step,
        model,
        valid_loaders,
        device,
        eval_iters=None,
        msg=None,
    ):
        """Evaluate the model on the validation data.

        Args:
            global_step (int): The training iteration number.
            model: A `torch.nn.Module` object.
            valid_loaders: A list of validation loaders, one
                for each action class.
            device: A `torch.device` object.
            eval_iters (int): The number of time to call
                `next` on the data iterator. Set to `None`
                to evaluate on the whole validation set.
            msg (str): An extra message to display in the
                progress bar. Use it to specify which split
                (train, valid) is being evaluated.

        Returns:
            A dict containing the evaluation metric
            results which can be scalar and/or images.
        """
        model.eval()

        # figure out a max batch size that's
        # a multiple of the number of context
        # frames.
        # this is so we can support large videos
        # with many frames.
        # Ideally, we'd want to move this inside
        # the model's forward pass and make it
        # adapt dynamically to the given device's memory.
        lcm = model.num_ctx_frames
        max_batch_size = math.floor(128 / lcm) * lcm

        # loop through every action class and compute
        # the metric.
        scalars, images = [], []
        pbar = tqdm(valid_loaders.items(), leave=False)
        for action_name, valid_loader in pbar:
            msg_ = "evaluating {}".format(action_name)
            if msg is not None:
                msg_ = "({}) ".format(msg) + msg_
            pbar.set_description(msg_)

            # check if the data loader is empty.
            # this is temporary since we don't have
            # labels for all videos and all class
            # actions.
            if len(valid_loader) == 0:
                logging.info(
                    "Skipping {} since it is empty.".format(action_name))
                continue

            # compute embeddings
            embeddings, labels, videos = [], [], []
            recons = []
            for batch_idx, batch in enumerate(valid_loader):
                if eval_iters is not None and batch_idx >= eval_iters:
                    break
                frames = batch["frames"]
                videos.append(frames.cpu())
                if frames.shape[1] > max_batch_size:
                    embs = []
                    for i in range(math.ceil(frames.shape[1] / max_batch_size)):
                        sub_frames = frames[
                            :, i * max_batch_size : (i + 1) * max_batch_size
                        ].to(device)
                        sub_embs = model(sub_frames)["embs"]
                        embs.append(sub_embs.cpu())
                    embs = torch.cat(embs, dim=1)
                else:
                    out = model(frames.to(device))
                    embs = out["embs"]
                    if "reconstruction" in out:
                        recons.append(out["reconstruction"].cpu())
                embeddings.append(embs.cpu().squeeze().numpy())
                # if "phase_idxs" in batch:
                if "debris_nums" in batch:
                    # labels.append(batch["phase_idxs"][0])
                    labels.append(batch["debris_nums"][0])
            if not labels:
                labels = None
            res = self._evaluate(
                embeddings, labels, videos, fit=(msg == "train"), recons=recons)
            if "scalar" in res:
                scalars.append(res["scalar"])
            if "image" in res:
                images.append(res["image"])

        # take the mean of the metric over classes
        ret = {}
        if scalars:
            if isinstance(scalars[0], dict):
                ret["scalar"] = dict_mean(scalars)
            else:  # list
                ret["scalar"] = np.mean(scalars)
        if images:
            ret["image"] = images
        return ret
