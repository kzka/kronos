"""Visualize TCC embeddings in 2D.
"""

import argparse
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import umap

from ipdb import set_trace
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from kronos.utils import file_utils

IMG_HEIGHT = 40
IMG_WIDTH = 40


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l)
        for i, (h, l) in enumerate(zip(handles, labels))
        if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def get_idxs(start, stop, count):
    step = (stop - start) / float(count)
    return np.asarray([start + i * step for i in range(count)]).astype(int)


def get_embs(embs_path, embodiment_names, num_traj):
    emb_files = file_utils.get_files(
        osp.join(embs_path, "embs"), "*.npy", sort=False
    )
    embs, frames, pos_neg_labels = {}, {}, {}
    for i, emb_file in enumerate(emb_files):
        emb_name = osp.basename(emb_file).split(".")[0]
        if emb_name not in embodiment_names:
            continue
        with open(emb_file, "rb") as f:
            data = np.load(f, allow_pickle=True).item()
        embs[emb_name] = data["embs"][:num_traj]
        frames[emb_name] = data["frames"][:num_traj]
        pos_neg_labels[emb_name] = data["labels"][:num_traj]
    return embs, frames, pos_neg_labels


def main(args):
    base_colors = ["b", "g", "r", "c", "m", "y", "k"]
    embodiment_names = [
        "two_hands_two_fingers",
        "tongs",
        "rms",
        # "ski_gloves",
        # "quick_grip",
        "one_hand_two_fingers",
        "one_hand_five_fingers",
        # "double_quick_grip",
        # "crab",
        # "quick_grasp",
    ]
    N = len(embodiment_names)

    embs, frames, pos_neg_labels = get_embs(
        args.embs_path, embodiment_names, args.num_traj
    )

    embeddings, rgb_frames, labels = [], [], []
    for name in embodiment_names:
        embeddings.extend(embs[name])
        rgb_frames.extend(frames[name])
        labels.extend(pos_neg_labels[name])
    if args.l2_normalize:
        embeddings = [x / (np.linalg.norm(x) + 1e-7) for x in embeddings]

    # figure out the min embedding length
    min_len = 10000000
    for emb in embeddings:
        if len(emb) < min_len:
            min_len = len(emb)
    print(f"min seq len: {min_len}")

    # subsample embedding sequences to make them
    # the same long since TSNE expects a tensor
    # of fixed sequence length
    embs, frames = [], []
    for i, (emb, rgb) in enumerate(zip(embeddings, rgb_frames)):
        idxs = get_idxs(0, len(emb), min_len)
        embs.append(emb[idxs])
        frames.append(rgb[idxs])
    embs = np.stack(embs)
    frames = np.stack(frames)

    # flatten embeddings (N, 128)
    num_vids, num_frames, num_feats = embs.shape
    embs_flat = embs.reshape(-1, num_feats)

    # dimensionality reduction
    if args.reducer == "umap":
        reducer = umap.UMAP(n_components=args.ndims, random_state=0)
    elif args.reducer == "tsne":
        reducer = TSNE(n_components=args.ndims, n_jobs=-1, random_state=0)
    elif args.reducer == "pca":
        reducer = PCA(n_components=args.ndims, random_state=0)
    else:
        raise ValueError(f"{args.reducer} is not a valid reducer.")

    embs_2d = reducer.fit_transform(embs_flat)

    # TODO: add variance calculation

    # subsample data for less cluttered visualization
    idxs = np.arange(args.num_traj * N)
    mask = []
    for idx in idxs:
        mask.extend(np.arange(idx * min_len, (idx + 1) * min_len))
    mask = np.asarray(mask)
    images = frames[idxs].reshape(-1, *frames.shape[2:])
    embs = embs_2d[mask]

    # resize frames
    frames = []
    for img in images:
        img = (img - img.min()) / (img.max() - img.min())
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.thumbnail((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
        frames.append(np.asarray(im))
    frames = np.stack(frames)

    label_names, colors = [], []
    for i, name in enumerate(embodiment_names):
        label_names.extend([name] * args.num_traj)
        colors.extend([base_colors[i]] * args.num_traj)

    # create figure
    fig = plt.figure()
    if args.ndims == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")

    lines, imgz, abz = [], [], []
    for i in range(len(idxs)):
        if args.ndims == 3:
            line = ax.scatter(
                embs[i * min_len : (i + 1) * min_len, 0],
                embs[i * min_len : (i + 1) * min_len, 1],
                embs[i * min_len : (i + 1) * min_len, 2],
                marker="o" if labels[i] else "x",
                s=15,
                c=colors[i],
                label=label_names[i],
            )
        else:
            line = ax.scatter(
                embs[i * min_len : (i + 1) * min_len, 0],
                embs[i * min_len : (i + 1) * min_len, 1],
                marker="o" if labels[i] else "x",
                s=15,
                c=colors[i],
                label=label_names[i],
            )
        lines.append(line)

        # create annotation box
        im = OffsetImage(frames[i * min_len], zoom=5)
        imgz.append(im)
        xybox = (IMG_HEIGHT, IMG_WIDTH)
        ab = AnnotationBbox(
            im,
            (0, 0),
            xybox=xybox,
            xycoords="data",
            boxcoords="offset points",
            pad=0.3,
            arrowprops=dict(arrowstyle="->"),
        )
        abz.append(ab)

        # add annotation box to axes and make it invisible
        ax.add_artist(ab)
        ab.set_visible(False)

        def hover(event):
            # if the mouse is over the scatter points
            for j, line in enumerate(lines):
                if line.contains(event)[0]:
                    ind = line.contains(event)[1]["ind"][0]
                    w, h = fig.get_size_inches() * fig.dpi
                    ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
                    hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)
                    abz[j].xybox = (xybox[0] * ws, xybox[1] * hs)
                    abz[j].set_visible(True)
                    abz[j].xy = (
                        embs[j * min_len + ind, 0],
                        embs[j * min_len + ind, 1],
                    )
                    imgz[j].set_data(frames[j * min_len + ind])
                else:
                    abz[j].set_visible(False)
                fig.canvas.draw_idle()

        # add callback for mouse move
        fig.canvas.mpl_connect("motion_notify_event", hover)
    legend_without_duplicate_labels(ax)
    emb_names = "_".join(embodiment_names)
    task_name = args.embs_path.split("/")[-2]
    name = "{}_{}_{}_{}_{}dim.png".format(
        task_name, emb_names, args.num_traj, args.reducer, args.ndims,
    )
    plt.savefig(osp.join("tmp/plots/", name), format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize video embeddings in a 2D plot."
    )
    parser.add_argument(
        "--embs_path", type=str, required=True, help="Path to the embeddings."
    )
    parser.add_argument(
        "--num_traj", type=int, required=True,
    )
    parser.add_argument(
        "--ndims", type=int, required=True, default=2,
    )
    parser.add_argument(
        "--l2_normalize",
        type=lambda s: s.lower() in ["true", "1"],
        default=False,
        help="Whether to l2 normalize embeddings before aligning.",
    )
    parser.add_argument(
        "--reducer", type=str, default="tsne",
    )
    args = parser.parse_args()
    main(args)
