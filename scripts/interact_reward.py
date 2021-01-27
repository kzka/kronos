import argparse
import os
import pdb
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gym
import albumentations as alb

from magical import wrappers
from magical.benchmarks import register_envs
from magical.env_interactor import KeyboardEnvInteractor

from kronos.utils import experiment_utils, checkpoint
from kronos.config import CONFIG

ENV_NAME = 'MatchRegions-Test{}-v0'
L2_NORMALIZE = True
EXPERIMENT_NAME = 'again_l2'


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
        start_emb = start_emb / np.linalg.norm(
            start_emb, axis=-1, keepdims=True)
        goal_emb = goal_emb / np.linalg.norm(
            goal_emb, axis=-1, keepdims=True)
    dists = np.linalg.norm(embs - goal_emb, axis=-1)
    # dist_emb_goal = 1 - (embs @ goal_emb.T)
    # dist_start_goal = 1 - (start_emb @ goal_emb.T)
    # dist_emb_goal = np.linalg.norm(embs - goal_emb, axis=-1)
    # dist_start_goal = np.linalg.norm(start_emb - goal_emb)
    # reward = 1 - (dist_emb_goal / dist_start_goal)
    reward = -dists
    return reward


@torch.no_grad()
def compute_goal_embedding(embedder, loaders, device):
    # goal_embs = []
    # start_embs = []
    for class_name, class_loader in loaders['downstream_train'].items():
        print(f"Embedding {class_name}.")
        for batch_idx, batch in enumerate(class_loader):
            if batch_idx >= 1:
                break
            # if batch_idx >= 200:
                # break
            # if batch_idx % 100 == 0:
                # print(f'\tEmbedding batch: {batch_idx}...')
            frames = batch["frames"]
            embs = embedder(frames.to(device))["embs"]
            embs_np = embs.cpu().squeeze().numpy()
            # start_embs.append(embs_np[0, :])
            # goal_embs.append(embs_np[-1, :])
    # start_emb = np.stack(start_embs, axis=0)
    # if L2_NORMALIZE:
    #     start_emb /= np.linalg.norm(start_emb, axis=-1, keepdims=True)
    # goal_emb = np.stack(goal_embs, axis=0)
    # if L2_NORMALIZE:
    #     goal_emb /= np.linalg.norm(goal_emb, axis=-1, keepdims=True)
    # start_emb = np.mean(start_emb, axis=0, keepdims=True)
    # goal_emb = np.mean(goal_emb, axis=0, keepdims=True)

    with open('./start_emb.pkl', 'rb') as fp:
        start_emb = pickle.load(fp)
    with open('./goal_emb.pkl', 'rb') as fp:
        goal_emb = pickle.load(fp)
    return start_emb, goal_emb


def setup_embedder(experiment_name, config_path):
    log_dir = os.path.join(CONFIG.DIRS.LOG_DIR, experiment_name)
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


def main(args):
    viewer = KeyboardEnvInteractor()
    # Load representation model and compute end-point embeddings.
    embedder, loaders = setup_embedder(EXPERIMENT_NAME, None)
    device = torch.device('cuda')
    embedder.to(device)
    embedder.eval()
    start_emb, goal_emb = compute_goal_embedding(embedder, loaders, device)
    # Load the environment.
    register_envs()
    env = gym.make(ENV_NAME.format(args.agent.capitalize()))
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
    plt.close()


def record_episode(seed, env, viewer, embedder, start_emb, goal_emb, device):
    # Render the first observation.
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
