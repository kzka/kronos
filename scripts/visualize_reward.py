import argparse
import math
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree as KDTree
from dtw import dtw

from kronos.utils import file_utils
from ipdb import set_trace


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
            nn_frame_id, nn_dist = get_nn(candidate_feats, query_feats[i])
            nns.append(nn_frame_id)
    return np.asarray(nns)


def get_nn(embs, query_emb):
    """Get the nearest-neighbour, i.e. smallest l2 distance.
    """
    dist = np.linalg.norm(embs - query_emb, axis=1)
    assert len(dist) == len(embs)
    return np.argmin(dist), np.min(dist)


def symmetric_chamfer(emb_query, emb_cand, kd_tree_query=None):
    if kd_tree_query is None:
        kd_tree_query = KDTree(emb_query)
    one_distances, _ = kd_tree_query.query(emb_cand)
    cand_to_query_chamfer = np.mean(np.square(one_distances))
    kd_tree_cand = KDTree(emb_cand)
    one_distances, _ = kd_tree_cand.query(emb_query)
    query_to_cand_chamfer = np.mean(np.square(one_distances))
    return cand_to_query_chamfer + query_to_cand_chamfer


def dense_reward(emb_query, emb_candidate, use_dtw):
    nns = align(emb_query, emb_candidate, use_dtw)
    nns = nns / len(emb_candidate)  # normalize between [0, 1]
    return nns


def modify_emb(emb, stride=2):
    seq_len = len(emb)
    return np.concatenate(
        [
            emb[0 : seq_len // 2 : stride],
            np.repeat(emb[seq_len // 2 :], stride, axis=0),
        ]
    )


def main(args):
    # get a list of embedding files
    # one for each action class
    emb_files = file_utils.get_files(
        osp.join(args.embs_path, "embs"), "*.npy", sort=False
    )
    print(f"Found {len(emb_files)} files.")

    all_embs, all_names = [], []
    for i, emb_file in enumerate(emb_files):
        embs = []
        with open(emb_file, "rb") as f:
            query_dict = np.load(f, allow_pickle=True).item()
        for j in range(len(query_dict["embs"])):
            curr_embs = query_dict["embs"][j]
            if args.l2_normalize:
                curr_embs = [x / (np.linalg.norm(x) + 1e-7) for x in curr_embs]
            embs.append(curr_embs)
        all_embs.append(embs)
        all_names.append(osp.basename(emb_file).split(".")[0])

    query_name = "rms"
    query_idx = all_names.index(query_name)
    query_embs = all_embs[query_idx]
    all_embs.pop(query_idx)
    all_names.pop(query_idx)

    # pick a random query trajectory
    emb_query = query_embs[2]  # np.random.choice(query_embs)
    kd_tree_query = KDTree(emb_query)

    # loop through all other embeddings
    # and find the closest trajectory
    # and its class
    # chamfer_dists = []
    class_min = -1
    traj_min = -1
    chamfer_dist = 100
    for i, emb_cands in enumerate(all_embs):
        for t, emb_cand in enumerate(emb_cands):
            dist = symmetric_chamfer(emb_query, emb_cand, kd_tree_query)
            if dist < chamfer_dist:
                class_min = i
                traj_min = t
                chamfer_dist = dist
    # chamfer_dists = np.array(chamfer_dists)
    # class_min = chamfer_dists.min(axis=1).argmin()
    # traj_min = chamfer_dists.argmin(axis=1)[class_min]
    print(all_names[class_min], traj_min)

    # artificially increase / decrease some trajectories
    # emb_best = all_embs[class_min][traj_min]
    emb_bad = all_embs[all_names.index("hand_5_fingers")][-9]
    # emb_best_mod = modify_emb(emb_best)
    # emb_bad_mod = modify_emb(emb_bad)

    # compute dense reward between query emb
    # and closest one as determined above
    # reward_best = dense_reward(emb_query, emb_best, True)
    reward_bad = dense_reward(emb_query, emb_bad, False)
    # reward_best_mod = dense_reward(emb_query, emb_best_mod)
    # reward_bad_mod = dense_reward(emb_query, emb_bad_mod)

    plt.figure()
    # plt.plot(reward_best, label='best')
    plt.plot(reward_bad, label="random")
    # plt.plot(reward_best_mod, label='best-updownsampled')
    # plt.plot(reward_bad_mod, label='random-updownsampled')
    plt.xlabel("Frame Index")
    plt.xlabel("Reward")
    plt.legend()
    plt.savefig("./reward_rms.png", format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embs_path", type=str, required=True, help="Path to the embeddings."
    )
    parser.add_argument(
        "--l2_normalize",
        type=lambda s: s.lower() in ["true", "1"],
        default=False,
        help="Whether to l2 normalize embeddings before aligning.",
    )
    args = parser.parse_args()
    main(args)
