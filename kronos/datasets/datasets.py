"""Base video dataset class.
"""

import logging
import os.path as osp
import random

import numpy as np
import torch

from torch.utils.data import Dataset

from kronos.utils import file_utils
from kronos.datasets import tensorizers, transforms
from ipdb import set_trace


class VideoDataset(Dataset):
    """A dataset for working with videos.
    """

    def __init__(
        self,
        root_dir,
        frame_sampler,
        labeled=False,
        augment_params=None,
        image_size=None,
        seed=None,
    ):
        """Constructor.

        Args:
            root_dir (str): The path to the dataset directory.
            frame_sampler (FrameSampler): A sampler specifying
                the frame sampling strategy.
            labeled (bool): Whether to load frame labels if they
                exist. These will usually be sparse phase action
                labels.
            augment_params (dict): A dict containing data
                augmentation hyper-parameters. Set to `None`
                to disable data augmentation.
            image_size (tuple): What size the loaded video
                frames should be.
            seed (int): The seed for the rng.

        Raises:
            ValueError: If the root directory is empty.
        """
        super().__init__()

        self._root_dir = root_dir
        self._frame_sampler = frame_sampler
        self._labeled = labeled
        self._seed = seed
        if augment_params is not None:
            # data augmentation is only applied on the video frames
            self._augmentor = transforms.Augmentor(
                "frames", augment_params, image_size
            )
        else:
            self._augmentor = None
        self._totensor = tensorizers.ToTensor()

        # seed the RNG
        self.seed_rng()

        # get list of available dirs and check that it is not empty
        self._available_dirs = file_utils.get_subdirs(
            self._root_dir, nonempty=True
        )
        if len(self._available_dirs) == 0:
            raise ValueError("{} is an empty directory.".format(root_dir))
        self._allowed_dirs = self._available_dirs

        # build a directory tree
        self._build_dir_tree()

    def seed_rng(self):
        if self._seed:
            random.seed(self._seed)

    def _build_dir_tree(self):
        """Build a dict of indices for iterating over the dataset.

        If labeled is not `None`, loads all subdirs. If labeled
        is `pos`, loads only positive demonstrations, and if
        labeled is `neg`, loads negative demonstrations.
        """
        self._dir_tree = {}
        for path in self._allowed_dirs:
            if self._labeled is not None:
                vids_labeled = []
                for vid in file_utils.get_subdirs(path, nonempty=False):
                    label_filename = osp.join(vid, "label.json")
                    if osp.exists(label_filename):
                        label = file_utils.load_json(label_filename)
                        if self._labeled == "pos" and label["success"] == 1:
                            vids_labeled.append(vid)
                        elif self._labeled == "neg" and label["success"] == 0:
                            vids_labeled.append(vid)
                        elif self._labeled == "both":
                            vids_labeled.append(vid)
                        else:
                            pass
                if len(vids_labeled) > 0:
                    self._dir_tree[path] = vids_labeled
            else:
                vids = file_utils.get_subdirs(path, nonempty=False)
                if len(vids) > 0:
                    self._dir_tree[path] = vids

    def restrict_subdirs(self, subdirs):
        """Restrict the set of available subdirectories.

        If using a batch sampler in conjunction with
        a dataloader, ensure `restrict_subdirs` is called
        before instantiating the sampler.

        Args:
            subdirs (list): A list of allowed video classes.

        Raises:
            ValueError: If the restriction leads to an empty directory.
        """
        if not isinstance(subdirs, list):
            subdirs = [subdirs]
        if subdirs:
            len_init = len(self._available_dirs)
            self._allowed_dirs = self._available_dirs
            subdirs = [osp.join(self._root_dir, x) for x in subdirs]
            self._allowed_dirs = list(set(self._allowed_dirs) & set(subdirs))
            if len(self._allowed_dirs) == 0:
                raise ValueError(
                    f"Filtering with {subdirs} returns an empty dataset."
                )
            len_final = len(self._allowed_dirs)
            logging.debug(
                f"Restricted dataset from {len_init} to {len_final} actions."
            )
            self._build_dir_tree()  # rebuild tree
        else:
            logging.debug("Passed in an empty list. No action taken.")

    def _get_video_paths(self, class_idx, vid_idxs):
        """Return video paths given class and video indices.

        For algorithms that operate on multiple videos
        (e.g. TCN operates on multi-view videos), `vid_idx`
        will be a list of video indices.

        Args:
            class_idx (int): The index of the action class
                folder in the dataset directory tree.
            vid_idxs (list or int): The index(s) of the
                video(s) in the action class folder to
                retrieve.

        Returns:
            A list of paths specifying which videos
            to sample in the dataset.
        """
        action_class = list(self._dir_tree)[class_idx]
        if not isinstance(vid_idxs, list):
            vid_idxs = [vid_idxs]
        vid_paths = []
        for vid_idx in vid_idxs:
            vid_paths.append(self._dir_tree[action_class][vid_idx])
        return vid_paths

    def _get_label(self, vid_paths):
        """Load the label json file.
        """
        assert len(vid_paths) == 1
        filename = osp.join(vid_paths[0], "label.json")
        return file_utils.load_json(filename)

    def _get_phase_label(self, vid_paths):
        assert len(vid_paths) == 1
        filename = osp.join(vid_paths[0], "phase_label.json")
        label = file_utils.load_json(filename)
        return label["phase_indices"]

    def _get_data(self, vid_paths):
        """Load video data given video paths.

        Feeds the video paths to the frame sampler
        to retrieve video frames and metadata.

        Args:
            vid_paths (str): A list of paths to videos in the
                dataset.

        Returns:
            A dictionary containing key, value pairs
            where the key is an enum member of `SequenceType`
            and the value is an ndarray respecting the key type.
        """
        # Phase indices only exist for successfully executed demonstrations.
        phase_indices = None
        if self._labeled == "pos":
            phase_indices = self._get_phase_label(vid_paths)
        sample = self._frame_sampler.sample(vid_paths, phase_indices)

        # Load debris state.
        filename = osp.join(vid_paths[0], "debris.json")
        debris_nums = file_utils.load_json(filename)
        debris_nums = np.array([debris_nums[0]] + debris_nums)
        debris_nums = debris_nums[sample["frame_idxs"]]

        # load each frame along with its context
        # frames into an array of shape
        # `(S, X, H, W, C)`, where `S` is the
        # number of sampled frames and `X`
        # is the number of context frames
        frames = np.stack([file_utils.load_jpeg(f) for f in sample["frames"]])
        frames = np.take(frames, sample["ctx_idxs"], axis=0)

        # reshape frames into a 4D array
        # of shape `(S * X, H, W, C)`
        frames = np.reshape(frames, (-1, *frames.shape[2:]))

        ret = {
            "frames": frames,
            "frame_idxs": np.asarray(sample["frame_idxs"], dtype=np.int64),
            "video_name": vid_paths,
            "video_len": sample["vid_len"],
            "debris_nums": debris_nums,
        }

        if sample["frame_labels"] is not None:
            ret["phase_labels"] = np.asarray(
                sample["frame_labels"], dtype=np.int64
            )

        # add label data if it exists
        if self._labeled is not None:
            label = self._get_label(vid_paths)
            ret["success"] = label["success"]

        return ret

    def __getitem__(self, idxs):
        vid_paths = self._get_video_paths(*idxs)
        data_np = self._get_data(vid_paths)
        if self._augmentor:
            data_np = self._augmentor(data_np)
        data_tensor = self._totensor(data_np)
        return data_tensor

    def __len__(self):
        return self.total_vids

    @property
    def num_classes(self):
        """The number of subdirs, i.e. allowed video classes.
        """
        return len(self._allowed_dirs)

    @property
    def total_vids(self):
        """The total number of videos across all allowed video classes.
        """
        num_vids = 0
        for vids in self._dir_tree.values():
            num_vids += len(vids)
        return num_vids

    @property
    def dir_tree(self):
        """The directory tree.
        """
        return self._dir_tree

    def collate_fn(self, batch):
        ret = {
            "frames": torch.stack([b["frames"] for b in batch]),
            "frame_idxs": torch.stack([b["frame_idxs"] for b in batch]),
            "video_len": torch.stack([b["video_len"] for b in batch]),
            "video_name": [b["video_name"] for b in batch],
            "debris_nums": torch.stack([b["debris_nums"] for b in batch]),
        }
        if self._labeled is not None:
            if "phase_labels" in batch[0]:
                ret["phase_labels"] = torch.stack(
                    [b["phase_labels"] for b in batch]
                )
            ret["success"] = [b["success"] for b in batch]
        return ret
