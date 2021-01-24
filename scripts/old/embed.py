"""Embed a video dataset using a trained temporal learning model.
"""

import argparse
import logging
import math

import numpy as np
import os.path as osp
import tqdm
import torch

from kronos.config import CONFIG
from kronos.utils import experiment_utils, file_utils, checkpoint
from kronos.datasets.transforms import UnNormalize
from ipdb import set_trace


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(
            "Using GPU {}.".format(torch.cuda.get_device_name(device))
        )
    else:
        logging.info("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    # initialize experiment
    opts = [
        "SAMPLING.STRIDE_ALL_SAMPLER",
        args.stride,
    ]
    config_path = osp.join(args.logdir, "config.yml")
    config, device = experiment_utils.init_experiment(
        args.logdir, CONFIG, config_path, opts,
    )

    # load model and data loaders
    debug = {
        "sample_sequential": not args.shuffle,
        "augment": False,
        "labeled": None,  # "both" if args.negative else "pos",
    }
    model, _, loaders, _, _ = experiment_utils.get_factories(
        config, device, debug=debug
    )
    loaders = (
        loaders["downstream_valid"]
        if args.split == "valid"
        else loaders["downstream_train"]
    )

    # load model checkpoint
    if args.model_ckpt is not None:
        checkpoint.Checkpoint(model).restore(args.model_ckpt, device)
    else:
        checkpoint.CheckpointManager.load_latest_checkpoint(
            checkpoint.Checkpoint(model),
            osp.join(
                config.DIRS.CKPT_DIR, osp.basename(osp.normpath(args.logdir))
            ),
            device,
        )
    model.to(device).eval()

    # figure out max batch size that's
    # a multiple of the number of context
    # frames.
    # this is so we can support large videos
    # with many frames.
    lcm = model.num_ctx_frames
    max_batch_size = math.floor(128 / lcm) * lcm

    # create save folder
    save_path = osp.join(
        config.DIRS.DIR,
        args.save_path,
        osp.basename(osp.normpath(args.logdir)),
        "embs",
    )
    file_utils.mkdir(save_path)

    # iterate over every class action
    pbar = tqdm.tqdm(loaders.items(), leave=False)
    for action_name, loader in pbar:
        msg = "embedding {}".format(action_name)
        pbar.set_description(msg)

        (
            embeddings,
            seq_lens,
            steps,
            vid_frames,
            names,
            labels,
            phase_labels,
        ) = ([] for i in range(7))
        for batch_idx, batch in enumerate(loader):
            if args.max_embs != -1 and batch_idx >= args.max_embs:
                break

            # unpack batch data
            frames = batch["frames"]
            chosen_steps = batch["frame_idxs"].to(device)
            seq_len = batch["video_len"].to(device)
            name = batch["video_name"][0]
            # label = batch["success"][0]
            # phase_label = None
            # if "phase_labels" in batch:
            #     phase_label = batch["phase_labels"].to(device)

            # forward through model to compute embeddings
            with torch.no_grad():
                if frames.shape[1] > max_batch_size:
                    embs = []
                    for i in range(
                        math.ceil(frames.shape[1] / max_batch_size)
                    ):
                        sub_frames = frames[
                            :, i * max_batch_size : (i + 1) * max_batch_size
                        ].to(device)
                        sub_embs = model(sub_frames)["embs"]
                        embs.append(sub_embs.cpu())
                    embs = torch.cat(embs, dim=1)
                else:
                    embs = model(frames.to(device))["embs"]

            # store
            embeddings.append(embs.cpu().squeeze().numpy())
            seq_lens.append(seq_len.cpu().squeeze().numpy())
            steps.append(chosen_steps.cpu().squeeze().numpy())
            # if phase_label is not None:
            # phase_labels.append(phase_label.cpu().squeeze().numpy())
            names.append(name)
            # labels.append(label.item())
            if args.keep_frames:
                frames = frames[0]
                frames = UnNormalize()(frames)
                frames = frames.view(
                    embs.shape[1],
                    config.SAMPLING.NUM_CONTEXT_FRAMES,
                    *frames.shape[1:],
                )
                vid_frames.append(
                    frames.cpu().squeeze().numpy()[:, -1].transpose(0, 2, 3, 1)
                )

        data = {
            "embs": embeddings,
            "seq_lens": seq_lens,
            "steps": steps,
            "names": names,
            # "labels": labels,
        }
        if args.keep_frames:
            data["frames"] = vid_frames
        # if phase_labels:
        # data["phase_labels"] = phase_labels
        np.save(osp.join(save_path, action_name), data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract embeddings from a Video Dataset."
    )
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid"],
        required=True,
        help="Which dataset split to load.",
    )
    parser.add_argument(
        "--max_embs",
        type=int,
        default=9,  # 3x3 grid
        help="The max number of videos to embed. -1 means embed all.",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="The spacing between the frames."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="tmp",
        help="The path to the saved embedding dicts.",
    )
    parser.add_argument(
        "--keep_frames",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Whether to store the video frames in the saved embedding dict.",
    )
    parser.add_argument(
        "--shuffle",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Whether to shuffle the videos.",
    )
    parser.add_argument(
        "--negative",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Whether to also embed negative trajectories.",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Which model checkpoint to load. By default, loads the "
        "checkpoint corresponding to the largest iter.",
    )
    args = parser.parse_args()
    main(args)
