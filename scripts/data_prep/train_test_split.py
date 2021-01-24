"""Splits a dataset into train-testing splits."""

import argparse
import logging
import numpy as np
import os.path as osp
import pdb

from glob import glob
from tqdm import tqdm

from kronos.utils import file_utils


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to videos.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where images will be stored."
        + " By default, stored in the input directory.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.975,
        help="The fraction used for train.",
    )
    args, unparsed = parser.parse_known_args()

    # Get list of subdirectories in the input dir
    input_subdirs = file_utils.get_subdirs(args.input_dir)

    # Create output dir if it does not exist.
    if args.output_dir is not None:
        file_utils.mkdir(args.output_dir)
        output_dir = args.output_dir
        operator = file_utils.copy_folder
    else:
        output_dir = args.input_dir
        operator = file_utils.move_folder

    for input_subdir in input_subdirs:
        video_dirs = file_utils.get_subdirs(input_subdir)

        # Create train and validation splits.
        np.random.shuffle(video_dirs)
        num_train = int(args.train_frac * len(video_dirs))
        train_files = video_dirs[:num_train]
        valid_files = video_dirs[num_train:]

        # Create directories.
        train_dir = osp.join(output_dir, osp.basename(input_subdir), "train")
        valid_dir = osp.join(output_dir, osp.basename(input_subdir), "valid")
        file_utils.mkdir(train_dir)
        file_utils.mkdir(valid_dir)

        # Apply operator on train and validation files.
        for f in train_files:
            dst = osp.join(train_dir, osp.basename(f))
            operator(f, dst)
        for f in valid_files:
            dst = osp.join(valid_dir, osp.basename(f))
            operator(f, dst)
