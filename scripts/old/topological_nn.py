"""Nearest topological neighbour.
"""

import argparse
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from ipdb import set_trace
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree as KDTree

from kronos.utils import file_utils


def symmetric_chamfer(emb_query, emb_cand, kd_tree_query=None):
    if kd_tree_query is None:
        kd_tree_query = KDTree(emb_query)
    one_distances, _ = kd_tree_query.query(emb_cand)
    cand_to_query_chamfer = np.mean(np.square(one_distances))
    kd_tree_cand = KDTree(emb_cand)
    one_distances, _ = kd_tree_cand.query(emb_query)
    query_to_cand_chamfer = np.mean(np.square(one_distances))
    return cand_to_query_chamfer + query_to_cand_chamfer


def get_idxs(start, stop, count):
    step = (stop - start) / float(count)
    return np.asarray([start + i * step for i in range(count)]).astype(int)


def main(args):
    emb_files = file_utils.get_files(
        osp.join(args.embs_path, "embs"), "*.npy", sort=False
    )
    print(f"Found {len(emb_files)} files.")

    embodiment_names = [
        "two_hands_two_fingers",
        "tongs",
        "rms",
        # "ski_gloves",
        # "quick_grip",
        "one_hand_two_fingers",
        "one_hand_five_fingers",
        "double_quick_grip",
        # "crab",
        # "quick_grasp",
    ]

    all_embs, all_names, phase_labels = [], [], []
    for i, emb_file in enumerate(emb_files):
        emb_name = osp.basename(emb_file).split(".")[0]
        if emb_name not in embodiment_names:
            continue
        embs = []
        with open(emb_file, "rb") as f:
            query_dict = np.load(f, allow_pickle=True).item()
        for j in range(len(query_dict["embs"])):
            if j == args.num_traj:
                break
            curr_embs = query_dict["embs"][j]
            if args.l2_normalize:
                curr_embs = [x / (np.linalg.norm(x) + 1e-7) for x in curr_embs]
            embs.append(curr_embs)
        all_embs.append(embs)
        all_names.append(emb_name)
        phase_labels.append(query_dict["phase_labels"])

    # split one_hand_five_fingers into 2
    h_idx = all_names.index("one_hand_five_fingers")
    h_emb = all_embs[h_idx]
    h_phase_labels = phase_labels[h_idx]
    all_names.pop(h_idx)
    all_embs.pop(h_idx)
    emb_one_hand_five_fivers_1 = []
    emb_one_hand_five_fivers_2 = []
    for emb, pl in zip(h_emb, h_phase_labels):
        if len(np.unique(pl)) == 2:
            emb_one_hand_five_fivers_1.append(emb)
        else:
            emb_one_hand_five_fivers_2.append(emb)
    all_names = all_names + [
        "one_hand_five_fingers_without_slide",
        "one_hand_five_fingers_with_slide",
    ]
    all_embs.append(emb_one_hand_five_fivers_1)
    all_embs.append(emb_one_hand_five_fivers_2)

    num_embs = len(all_embs)
    chamfers = np.zeros((num_embs, num_embs))
    for i in range(len(all_embs)):
        query_embs = all_embs[i]
        for j in range(len(all_embs)):
            if i == j:
                continue
            cand_embs = all_embs[j]
            query_chamfer = []
            for query_emb in query_embs:
                kd_tree_query = KDTree(query_emb)
                chamfer_dist = []
                for cand_emb in cand_embs:
                    chamfer_dist.append(
                        symmetric_chamfer(query_emb, cand_emb, kd_tree_query)
                    )
                query_chamfer.append(np.mean(chamfer_dist))
            chamfers[i, j] = np.mean(query_chamfer)

    # determine row-wise min
    mins = []
    for i in range(num_embs):
        mins.append(np.argsort(chamfers[i])[1])

    # plot heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(chamfers, cmap="RdBu", interpolation="nearest")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Chamfer Distance", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(num_embs))
    ax.set_yticks(np.arange(num_embs))
    ax.set_xticklabels(all_names)
    ax.set_yticklabels(all_names)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.xaxis.set_tick_params(labelsize=7)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    for i in range(num_embs):
        for j in range(num_embs):
            if i == j:
                continue
            txt = "{:.5f}".format(chamfers[i, j])
            if j == mins[i]:
                txt += "_min"
            _ = ax.text(
                j, i, txt, ha="center", va="center", color="black", fontsize=7
            )
    task_name = args.embs_path.split("/")[-2]
    ax.set_title(task_name)
    name = "{}_heatmap.png".format(task_name)
    plt.savefig(osp.join("tmp", name), format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embs_path", type=str, required=True, help="Path to the embeddings."
    )
    parser.add_argument(
        "--num_traj", type=int, default=2,
    )
    parser.add_argument(
        "--l2_normalize",
        type=lambda s: s.lower() in ["true", "1"],
        default=True,
        help="Whether to l2 normalize embeddings before aligning.",
    )
    args = parser.parse_args()
    main(args)
