"""Teleop the agent and visualize the learned reward.

$ python scripts/interact_reward.py \
    --agent gripper
"""

import albumentations as alb
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

from magical import wrappers
from magical.benchmarks import register_envs
from magical.env_interactor import KeyboardEnvInteractor

from kronos.utils import experiment_utils, checkpoint
from kronos.config import CONFIG

ENV_NAME = 'MatchRegions-Test{}-v0'
L2_NORMALIZE = True
EXPERIMENT_NAME = 'all_but_gripper_samebatch'  # 'all_but_gripper'  # 'again_l2'


@torch.no_grad()
def embed(obs, model, device):
    resize_transform = alb.Compose([alb.Resize(224, 224, p=1.0)])
    x = resize_transform(image=obs)["image"]
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, ...]
    x = x / 255.0
    x = x.to(device)
    out = model(x)["embs"]
    emb = out[0].detach().cpu().numpy()
    return emb


@torch.no_grad()
def compute_reward(obses, start_emb, goal_emb, embedder, device):
    embs = [embed(o, embedder, device) for o in obses]
    embs = np.concatenate(embs, axis=0)
    if L2_NORMALIZE:
        embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
    dists = np.linalg.norm(embs - goal_emb, axis=-1)
    # dist_emb_goal = 1 - (embs @ goal_emb.T)
    # dist_start_goal = 1 - (start_emb @ goal_emb.T)
    # dist_emb_goal = np.linalg.norm(embs - goal_emb, axis=-1)
    # dist_start_goal = np.linalg.norm(start_emb - goal_emb)
    # reward = 1 - (dist_emb_goal / dist_start_goal)
    reward = -dists
    return reward


def setup_embedder(log_dir, config_path):
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    config, device = experiment_utils.init_experiment(
        log_dir, CONFIG, config_path)
    embedder, _, _, _, _ = experiment_utils.get_factories(
        config, device, debug={'augment': False})
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(embedder), checkpoint_dir, device)
    global_step = checkpoint_manager.restore_or_initialize()
    print(f"Restored model from checkpoint {global_step}")
    return embedder


def load_embeddings(log_dir):
    try:
        with open(os.path.join(log_dir, 'start_emb.pkl'), 'rb') as fp:
            start_emb = pickle.load(fp)
        with open(os.path.join(log_dir, 'goal_emb.pkl'), 'rb') as fp:
            goal_emb = pickle.load(fp)
        return start_emb, goal_emb
    except FileNotFoundError:
        print("Make sure you have precomputed the embeddings.")


def main(args):
    log_dir = os.path.join(CONFIG.DIRS.LOG_DIR, EXPERIMENT_NAME)
    device = torch.device('cuda')
    embedder = setup_embedder(log_dir, None)
    embedder.to(device).eval()
    start_emb, goal_emb = load_embeddings(log_dir)
    register_envs()
    env = gym.make(ENV_NAME.format(args.agent.capitalize()))
    viewer = KeyboardEnvInteractor()
    run_episode(env, viewer, embedder, start_emb, goal_emb, device)


def run_episode(env, viewer, embedder, start_emb, goal_emb, device):
    seed = np.random.randint(0, 100)
    env.seed(seed)
    env.reset()
    record_episode(seed, env, viewer, embedder, start_emb, goal_emb, device)


def plot_reward(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.show()


def record_episode(seed, env, viewer, embedder, start_emb, goal_emb, device):
    # Render the first observation and display it.
    obs = env.render('rgb_array')
    viewer.imshow(obs)

    i = [1]
    obses = [obs]
    def step(action):
        _, rew, done, info = env.step(action)
        obs = env.render('rgb_array')
        obses.append(obs)
        if i[0] % 50 == 0:
            print(f"Done, score {info['eval_score']:.4g}/1.0")
        i[0] += 1
        return obs
    viewer.run_loop(step)

    rewards = compute_reward(obses, start_emb, goal_emb, embedder, device)
    plot_reward(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default='gripper')
    args = parser.parse_args()
    main(args)
