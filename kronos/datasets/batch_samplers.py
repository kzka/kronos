import abc
import logging
import os.path as osp

import torch

from torch.utils.data import Sampler

from kronos.utils import file_utils
from ipdb import set_trace


class VideoBatchSampler(Sampler):
    """Base class for all video samplers.

    Every `VideoBatchSampler` subclass has to provide an `__iter__()` method,
    providing a way to iterate over indices of dataset elements, and
    a `__len__()` method that returns the length of the returned iterators.
    """

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class RandomBatchSampler(VideoBatchSampler):
    """Randomly samples videos from different action classes into the same batch.
    """

    def __init__(self, dir_tree, batch_size, sequential=False, labeled=False):
        """Constructor.

        Args:
            dir_tree (dict): The directory tree of a `VideoDataset`.
            batch_size (int): The number of videos in a batch.
        """
        self._dir_tree = dir_tree
        self.batch_size = int(batch_size)

    def update_tree(self, dir_tree):
        """Update the directory tree.

        This should be used when a `VideoDataset`'s `restrict_subdirs` method
        is called to update the new directory tree.
        """
        self._dir_tree = dir_tree

    def _gen_idxs(self):
        """Generate the video indices.
        """
        # loop over every action class folder
        all_idxs = []
        for k, v in enumerate(self._dir_tree.values()):
            len_v = len(v)
            # generate a list of indices for every video
            # in the action class
            seq = list(range(len_v))
            idxs = [(k, s) for s in seq]
            all_idxs.extend(idxs)
        # shuffle the indices
        all_idxs = [all_idxs[i] for i in torch.randperm(len(all_idxs))]
        # split the list of indices into chunks of len
        # `batch_size`, ensuring we drop the last chunk
        # if it is not of adequate length
        self.idxs = []
        end = self.batch_size * (len(all_idxs) // self.batch_size)
        for i in range(0, end, self.batch_size):
            xs = all_idxs[i : i + self.batch_size]
            self.idxs.append(xs)

    def __iter__(self):
        self._gen_idxs()
        return (self.idxs[i] for i in torch.randperm(len(self.idxs)))

    def __len__(self):
        num_vids = 0
        for vids in self._dir_tree.values():
            num_vids += len(vids)
        return num_vids // self.batch_size


class SameClassBatchSampler(VideoBatchSampler):
    """Ensures all videos in a batch belong to the same action class.
    """

    def __init__(self, dir_tree, batch_size, sequential=False, labeled=False):
        """Constructor.

        Args:
            dir_tree (dict): The directory tree of a `VideoDataset`.
            batch_size (int): The number of videos in a batch.
            sequential (bool): Deterministically and sequentially
                sample video indices. This is useful for debugging
                a model, for example by overfitting on a specific
                video subset.
        """
        self._dir_tree = dir_tree
        self.batch_size = int(batch_size)
        self._sequential = sequential
        if self._sequential:
            logging.info("Batch sampler set to SEQUENTIAL.")
        else:
            logging.info("Batch sampler set to SHUFFLED.")
        self._labeled = labeled
        if self._labeled:
            logging.info("Balanced positive/negative sampling ENABLED.")

    def update_tree(self, dir_tree):
        """Update the directory tree.

        This should be used when a `VideoDataset`'s `restrict_subdirs` method
        is called to update the new directory tree.
        """
        self._dir_tree = dir_tree

    def _gen_idxs(self):
        """Generate the video indices.
        """
        # loop over every action class folder
        self.idxs = []
        for k, v in enumerate(self._dir_tree.values()):
            len_v = len(v)
            if self._labeled:
                # we need to figure out which videos
                # in the action class are positive
                # labeled, and which are negative
                # labeled
                pos_idxs, neg_idxs = [], []
                for i, vid in enumerate(v):
                    label_filename = osp.join(vid, "label.json")
                    label = file_utils.load_json(label_filename)
                    if label["success"]:
                        pos_idxs.append(i)
                    else:
                        neg_idxs.append(i)

                if not self._sequential:
                    pos_idxs = [
                        pos_idxs[i] for i in torch.randperm(len(pos_idxs))
                    ]
                    neg_idxs = [
                        neg_idxs[i] for i in torch.randperm(len(neg_idxs))
                    ]

                # interleave positive and negative indices
                seq = []
                for i in range(min(len(pos_idxs), len(neg_idxs))):
                    seq.extend([pos_idxs[i], neg_idxs[i]])
                if len(pos_idxs) > len(neg_idxs):
                    seq.extend(pos_idxs[len(neg_idxs) :])
                elif len(pos_idxs) < len(neg_idxs):
                    seq.extend(neg_idxs[len(pos_idxs) :])
                else:
                    pass
            else:
                # generate a list of indices for every video
                # in the action class
                seq = list(range(len_v))
                if not self._sequential:
                    seq = [seq[i] for i in torch.randperm(len(seq))]
            # split the list of indices into chunks of len
            # `batch_size`, ensuring we drop the last chunk
            # if it is not of adequate length
            idxs = []
            end = self.batch_size * (len_v // self.batch_size)
            for i in range(0, end, self.batch_size):
                xs = seq[i : i + self.batch_size]
                # add the action class index to the
                # video index
                xs = [(k, x) for x in xs]
                idxs.append(xs)
            self.idxs.extend(idxs)

    def __iter__(self):
        self._gen_idxs()
        # print(self.idxs)
        if not self._sequential:
            return (self.idxs[i] for i in torch.randperm(len(self.idxs)))
        return iter(self.idxs)

    def __len__(self):
        num_vids = 0
        for vids in self._dir_tree.values():
            num_vids += len(vids)
        return num_vids // self.batch_size


class SameClassBatchSamplerDownstream(SameClassBatchSampler):
    """A same class batch sampler with a batch size of 1.

    This batch sampler is used in downstream datasets. Since
    downstream datasets need to load a variable number of
    frames per video, we cannot load more than 1 video per
    batch.
    """

    def __init__(self, dir_tree, sequential=False, labeled=False):
        """Constructor.

        Args:
            dir_tree (dict): The directory tree of a `VideoDataset`.
            sequential (bool): Deterministically and sequentially
                sample video indices. This is useful for debugging
                a model, for example by overfitting on a specific
                video subset.
        """
        super().__init__(dir_tree, 1, sequential, labeled)


class SameClassMultiViewBatchSampler(VideoBatchSampler):
    """A same-class batch sampler that samples multiple videos
    of the same content but filmed from different views.
    """

    pass
