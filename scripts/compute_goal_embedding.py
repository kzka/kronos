"""Pre-compute the goal embedding using a trained model.

$ python scripts/compute_goal_embedding.py \
    --experiment_name all_but_gripper \
    --l2_normalize
"""

import argparse
import numpy as np
import os
import pickle
import torch

from kronos.utils import experiment_utils, checkpoint
from kronos.config import CONFIG


@torch.no_grad()
def embed(embedder, loaders, l2_normalize, device):
    """Embed the stored trajectories to get the start and goal embeddings."""
    goal_embs = []
    start_embs = []
    for class_name, class_loader in loaders['downstream_train'].items():
        print(f"Embedding {class_name}.")
        for batch_idx, batch in enumerate(class_loader):
            if batch_idx % 100 == 0:
                print(f'\tEmbedding batch: {batch_idx}...')
            frames = batch["frames"]
            embs = embedder(frames.to(device))["embs"]
            embs_np = embs.cpu().squeeze().numpy()
            start_embs.append(embs_np[0, :])
            goal_embs.append(embs_np[-1, :])
    start_emb = np.stack(start_embs, axis=0)
    goal_emb = np.stack(goal_embs, axis=0)
    if l2_normalize:
        start_emb /= np.linalg.norm(start_emb, axis=-1, keepdims=True)
        goal_emb /= np.linalg.norm(goal_emb, axis=-1, keepdims=True)
    start_emb = np.mean(start_emb, axis=0, keepdims=True)
    goal_emb = np.mean(goal_emb, axis=0, keepdims=True)
    return start_emb, goal_emb


def setup_embedder(log_dir, config_path):
    """Load the latest embedder checkpoint and dataloaders."""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    config, device = experiment_utils.init_experiment(
        log_dir, CONFIG, config_path)
    embedder, _, loaders, _, _ = experiment_utils.get_factories(
        config, device, debug={'augment': False})
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(embedder), checkpoint_dir, device)
    global_step = checkpoint_manager.restore_or_initialize()
    print(f"Restored model from checkpoint {global_step}")
    return embedder, loaders


def pickle_dump(filename, arr):
    """Dump as a pickle file."""
    with open(filename, 'wb') as fp:
        pickle.dump(arr, fp)


def main(experiment_name):
    device = torch.device('cuda')
    log_dir = os.path.join(CONFIG.DIRS.LOG_DIR, experiment_name)
    embedder, loaders = setup_embedder(log_dir, None)
    embedder.to(device).eval()
    l2_normalize = config.LOSS.L2_NORMALIZE_EMBEDDINGS
    if l2_normalize:
        print('L2 normalizing embeddings.')
    start_emb, goal_emb = embed(embedder, loaders, l2_normalize, device)
    pickle_dump(os.path.join(log_dir, 'start_emb.pkl'), start_emb)
    pickle_dump(os.path.join(log_dir, 'goal_emb.pkl'), goal_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()
    main(args.experiment_name)
