"""Process MIME videos into frames.
"""

import os.path as osp
import numpy as np

from ast import literal_eval
from collections import Counter
from distutils.dir_util import copy_tree
from tqdm import tqdm

from kronos.utils import file_utils
from kronos.datasets.action_class import mime_map


def chunk_means(arr, n):
    """Computes the mean of `n` contiguous chunks of an array.
  """
    means = []
    for i in range(0, len(arr), n):
        means.append(np.mean(arr[i : i + n]))
    return means


def majority(x):
    """Returns the most common element in `x`.
  """
    c = Counter(x)
    value, _ = c.most_common()[0]
    return value


def is_static(joint_angles, idx):
    """Check that a time series of joint angles is static.

  This is to determine which Baxter arm was the one that
  moved during the demonstration.
  """
    means = chunk_means(joint_angles[idx], 10)
    same = np.array(
        [
            np.isclose(means[i], means[i + 1], atol=1e-2)
            for i in range(len(means) - 1)
        ]
    )
    return len(same[same is False])


# taken from https://github.com/pathak22/hierarchical-imitation/blob/master/utils.py
def preprocess_jointangles(sequence):
    seq_array = []
    for index in range(0, len(sequence)):
        sub_seq = literal_eval(sequence[index])
        temp = np.zeros(14)
        k = [
            "left_w0",
            "left_w1",
            "left_w2",
            "right_s0",
            "right_s1",
            "right_w0",
            "right_w1",
            "right_w2",
            "left_e0",
            "left_e1",
            "left_s0",
            "left_s1",
            "right_e0",
            "right_e1",
        ]
        if len(sub_seq) > 14:
            for i in range(0, len(k)):
                temp[i] = sub_seq[k[i]]
        seq_array.append(np.copy(temp))
    return np.asarray(seq_array)


def load_joint_angles(dir):
    idxs_left = [8, 9]
    idxs_right = [12, 13]
    with open(osp.join(dir, "joint_angles.txt")) as f:
        joint_angles = f.read().splitlines()
    joint_angles = preprocess_jointangles(joint_angles).T
    num_false_right = np.mean(
        [
            is_static(joint_angles, idxs_right[0]),
            is_static(joint_angles, idxs_right[1]),
        ]
    )
    num_false_left = np.mean(
        [
            is_static(joint_angles, idxs_left[0]),
            is_static(joint_angles, idxs_left[1]),
        ]
    )
    if num_false_right < num_false_left:
        return ["RD_sk_right_rgb"]
    return ["RD_sk_left_rgb"]


if __name__ == "__main__":
    VALIDATION_FRAC = 0.1
    # set to False to include the static side view
    OVERHEAD_ONLY = True
    # whether to include depth videos
    INCLUDE_DEPTH = False
    in_dir = "/home/kevin/repos/kronos/kronos/data/mime/"
    out_dir = "/home/kevin/repos/kronos/kronos/data/mime_processed/"
    for folder_name, folder_id in mime_map.items():
        in_pouring_dir = osp.join(in_dir, "{}".format(folder_id))
        wanted_folders = ["hd_kinect_rgb", "rd_kinect_rgb"]
        if not OVERHEAD_ONLY:
            wanted_folders += ["rd_kinect_rgb"]
        demonstrations = file_utils.get_subdirs(in_pouring_dir)
        num_valid = int(VALIDATION_FRAC * len(demonstrations))
        valid_demonstrations = demonstrations[:num_valid]
        train_demonstrations = demonstrations[num_valid:]
        for name, demonstrations in zip(
            ["train", "valid"], [train_demonstrations, valid_demonstrations]
        ):
            out_pouring_dir = osp.join(
                out_dir, name, folder_name.lower().replace(" ", "_")
            )
            file_utils.mkdir(out_pouring_dir)
            cnter = 0
            for in_demo in tqdm(demonstrations):
                left_or_right = (
                    [] if OVERHEAD_ONLY else load_joint_angles(in_demo)
                )
                for wanted in wanted_folders + left_or_right:
                    in_wanted = osp.join(in_demo, wanted)
                    out_wanted = osp.join(out_pouring_dir, "{}".format(cnter))
                    if not osp.isdir(out_wanted):
                        copy_tree(in_wanted, out_wanted)
                    cnter += 1
