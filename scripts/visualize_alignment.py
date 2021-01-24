# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made by Kevin Zakka.

"""Align videos based on nearest neighbor in embedding space.
"""

import argparse
import math
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dtw import dtw
from matplotlib.animation import FuncAnimation

from kronos.utils import file_utils


def get_nn(embs, query_emb):
    """Get the nearest-neighbour, i.e. smallest l2 distance.
    """
    dist = np.linalg.norm(embs - query_emb, axis=1)
    assert len(dist) == len(embs)
    return np.argmin(dist), np.min(dist)


def unnorm(frame):
    """Unnormalize an image.
    """
    min_v = frame.min()
    max_v = frame.max()
    return (frame - min_v) / (max_v - min_v)


def align(query_feats, candidate_feats, use_dtw):
    """Align videos based on nearest neighbor or dynamic time warping.
    """
    if use_dtw:
        dist_fn = lambda x, y: np.sum((x - y) ** 2)  # noqa: E731
        _, _, _, path = dtw(query_feats, candidate_feats, dist=dist_fn)
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
    else:
        nns = []
        for i in range(len(query_feats)):
            nn_frame_id, _ = get_nn(candidate_feats, query_feats[i])
            nns.append(nn_frame_id)
    return nns


def create_original_videos(frames, video_path, interval):
    """Play each video at their original speed, i.e. alignment is turned off.
    """
    ncols = int(math.sqrt(len(frames)))
    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=ncols,
        figsize=(5 * ncols, 5 * ncols),
        tight_layout=True,
    )
    max_len = max([len(f) for f in frames])

    def init():
        ims = []
        k = 0
        for k in range(ncols):
            for j in range(ncols):
                ims.append(ax[j][k].imshow(unnorm(frames[k * ncols + j][0])))
                ax[j][k].grid(False)
                ax[j][k].set_xticks([])
                ax[j][k].set_yticks([])
        return ims

    ims = init()

    def update(i):
        print("{}/{}".format(i, max_len))
        for k in range(ncols):
            for j in range(ncols):
                idx = (
                    i
                    if i < len(frames[k * ncols + j])
                    else len(frames[k * ncols + j]) - 1
                )
                ims[k * ncols + j].set_data(unnorm(frames[k * ncols + j][idx]))
        plt.tight_layout()
        return ims

    anim = FuncAnimation(
        fig, update, frames=np.arange(max_len), interval=interval, blit=False,
    )
    anim.save(video_path, dpi=80)


def create_video(
    embs, frames, video_path, use_dtw, query, candidate, interval
):
    """Create aligned videos."""
    if candidate is not None:
        fig, ax = plt.subplots(ncols=2, figsize=(10, 10), tight_layout=True)
        nns = align(embs[query], embs[candidate], use_dtw)

        def init():
            ims = []
            ims.append(ax[0].imshow(unnorm(frames[query][0])))
            ims.append(ax[1].imshow(unnorm(frames[candidate][nns[0]])))
            for i in range(2):
                ax[i].grid(False)
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            return ims

        ims = init()

        def update(i):
            """Update plot with next frame."""
            print("{}/{}".format(i, len(embs[query])))
            ims[0].set_data(unnorm(frames[query][i]))
            ims[1].set_data(unnorm(frames[candidate][nns[i]]))
            plt.tight_layout()

    else:
        ncols = int(math.sqrt(len(embs)))
        print(
            "There are {} embeddings. Making a {}x{} plot.".format(
                len(embs), ncols, ncols
            )
        )
        fig, ax = plt.subplots(
            ncols=ncols,
            nrows=ncols,
            figsize=(5 * ncols, 5 * ncols),
            tight_layout=True,
        )
        nns = []
        for candidate in range(len(embs)):
            nns.append(align(embs[query], embs[candidate], use_dtw))

        def init():
            ims = []
            k = 0
            for k in range(ncols):
                for j in range(ncols):
                    ims.append(
                        ax[j][k].imshow(
                            unnorm(
                                frames[k * ncols + j][nns[k * ncols + j][0]]
                            )
                        )
                    )
                    ax[j][k].grid(False)
                    ax[j][k].set_xticks([])
                    ax[j][k].set_yticks([])
            return ims

        ims = init()

        def update(i):
            print("{}/{}".format(i, len(embs[query])))
            for k in range(ncols):
                for j in range(ncols):
                    ims[k * ncols + j].set_data(
                        unnorm(frames[k * ncols + j][nns[k * ncols + j][i]])
                    )
            plt.tight_layout()
            return ims

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(len(embs[query])),
        interval=interval,
        blit=False,
    )
    anim.save(video_path, dpi=80)


def main(args):
    plt.switch_backend("Agg")

    emb_files = file_utils.get_files(
        osp.join(args.embs_path, "embs"), "*.npy", sort=False
    )
    print("Found files: {}".format(emb_files))

    for i, emb_file in enumerate(emb_files):
        embs, frames = [], []
        with open(emb_file, "rb") as f:
            query_dict = np.load(f, allow_pickle=True).item()
        for j in range(len(query_dict["embs"])):
            curr_embs = query_dict["embs"][j]
            if args.l2_normalize:
                curr_embs = [x / (np.linalg.norm(x) + 1e-7) for x in curr_embs]
            embs.append(curr_embs)
            frames.append(query_dict["frames"][j])

        # generate video name
        video_path = osp.join(args.embs_path, "videos")
        file_utils.mkdir(video_path)
        ext = ".mp4" if args.align else "_original.mp4"
        video_path = osp.join(
            video_path, osp.basename(emb_file).split(".")[0] + ext
        )

        if not args.align:
            print("Playing videos without alignment.")
            create_original_videos(frames, video_path, args.interval)
        else:
            print("Aligning videos.")
            create_video(
                embs,
                frames,
                video_path,
                args.use_dtw,
                query=args.reference_video,
                candidate=args.candidate_video,
                interval=args.interval,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract embeddings from a Video Dataset."
    )
    parser.add_argument(
        "--embs_path", type=str, required=True, help="Path to the embeddings."
    )
    parser.add_argument(
        "--align",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Whether to align or play videos at original speed.",
    )
    parser.add_argument(
        "--use_dtw",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Whether to use dynamic time warping.",
    )
    parser.add_argument(
        "--reference_video",
        type=int,
        default=0,
        help="The reference video on which all others are aligned.",
    )
    parser.add_argument(
        "--candidate_video",
        type=int,
        default=None,
        help="The candidate video. If `None`, we align all videos.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Time in ms in b/w consecutive frames.",
    )
    parser.add_argument(
        "--l2_normalize",
        type=lambda s: s.lower() in ["true", "1"],
        default=False,
        help="Whether to l2 normalize embeddings before aligning.",
    )
    args = parser.parse_args()
    main(args)
