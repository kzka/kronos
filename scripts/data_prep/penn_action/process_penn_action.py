"""Restructure Penn Action dataset into class action folders.
"""

import argparse
import logging
import os.path as osp

from collections import OrderedDict
from multiprocessing import cpu_count, Pool, set_start_method
from scipy.io import loadmat
from tqdm import tqdm

from kronos.utils import file_utils


def resize_and_move_frame(args):
    """Resize and save a video frame.
  """
    frame_path, output_dir, resize = args
    frame = file_utils.load_jpeg(frame_path, resize)
    dst_path = osp.join(output_dir, osp.basename(frame_path))
    file_utils.write_image_to_jpeg(frame, dst_path)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to videos."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where images will be stored."
        + " By default, stored in the input directory.",
    )
    parser.add_argument(
        "--resize",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Resize the video frames to a specified size.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="The height of processed video frames if resize is set to True.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="The width of processed video frames if resize is set to True.",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=cpu_count(),
        help="The number of cores to use in parallel.",
    )
    args, unparsed = parser.parse_known_args()

    # create output dir if it doesn't exist
    if args.output_dir is not None:
        file_utils.mkdir(args.output_dir)
    else:
        args.output_dir = args.input_dir
    frame_dir = osp.join(args.input_dir, "frames")
    label_dir = osp.join(args.input_dir, "labels")

    if args.resize:
        args.resize = (args.height, args.width)
        logging.info("Resizing images to ({}, {}).".format(*args.resize))
    else:
        args.resize = None

    # read all label files and figure out which
    # videos belong to which class
    label_files = file_utils.get_files(label_dir, "*.mat")
    label_classes = [loadmat(lab)["action"][0] for lab in label_files]

    # figure out train test splits
    train_val_splits = [loadmat(lab)["train"][0][0] for lab in label_files]

    # now figure out at which indices the class
    # change occurs
    change_idxs = []
    for i in range(len(label_classes) - 1):
        c_curr = label_classes[i]
        c_next = label_classes[i + 1]
        if c_curr != c_next:
            change_idxs.append(i + 1)
    change_idxs = [0, *change_idxs, len(label_files)]

    # get unique label classes whilst preserving order
    label_classes = list(OrderedDict.fromkeys(label_classes).keys())
    num_classes = len(label_classes)

    set_start_method("spawn")
    with Pool(args.num_cores) as pool:
        for i in range(len(change_idxs) - 1):
            logging.info(
                "Processing {} ({}/{}).".format(
                    label_classes[i], i + 1, num_classes
                )
            )

            from_idx = change_idxs[i] + 1
            to_idx = change_idxs[i + 1] + 1
            train_val_split = train_val_splits[change_idxs[i] : to_idx]

            # create class folder
            vid_train_dir = osp.join(
                args.output_dir, "train", label_classes[i]
            )
            vid_valid_dir = osp.join(
                args.output_dir, "valid", label_classes[i]
            )
            file_utils.mkdir(vid_train_dir)
            file_utils.mkdir(vid_valid_dir)

            # copy videos belonging to class to the
            # new class folder
            for idx, j in enumerate(range(from_idx, to_idx)):
                vid_src = osp.join(frame_dir, "{:04d}".format(j))
                vid_dst = osp.join(
                    vid_train_dir
                    if train_val_split[idx] == 1
                    else vid_valid_dir,
                    "{:04d}".format(j),
                )
                frames = file_utils.get_files(vid_src, "*.jpg")

                # create video folder
                file_utils.mkdir(vid_dst)

                # move frames to new destination and optionally resize
                func_args = [[f, vid_dst, args.resize] for f in frames]
                for _ in tqdm(
                    pool.imap_unordered(resize_and_move_frame, func_args),
                    total=len(frames),
                    leave=False,
                ):
                    pass
