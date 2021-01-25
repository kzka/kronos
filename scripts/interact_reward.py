import argparse
import os
import pdb
import torch
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
def compute_reward(frame, goal_emb, embedder, device):
    emb = embed(frame, embedder, device)
    # emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    dist = -np.linalg.norm(emb - goal_emb)
    reward = dist
    return reward


@torch.no_grad()
def compute_goal_embedding(embedder, loaders, device):
    embeddings = []
    for class_name, class_loader in loaders['downstream_train'].items():
        print(f"Embedding {class_name}.")
        for batch_idx, batch in enumerate(class_loader):
            if batch_idx >= 100:
                break
            if batch_idx % 100 == 0:
                print(f'\tEmbedding batch: {batch_idx}...')
            frames = batch["frames"][:, -1:]
            embs = embedder(frames.to(device))["embs"]
            embeddings.append(embs.cpu().squeeze().numpy())
    goal_emb = np.stack(embeddings, axis=0)
    # goal_emb /= np.linalg.norm(goal_emb, axis=-1, keepdims=True)
    goal_emb = np.mean(goal_emb, axis=0, keepdims=True)
    return goal_emb


def setup_embedder(experiment_name, config_path):
    log_dir = os.path.join(CONFIG.DIRS.LOG_DIR, experiment_name)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')

    config, device = experiment_utils.init_experiment(
        log_dir, CONFIG, config_path)

    embedder, _, loaders, _, _ = experiment_utils.get_factories(config, device)

    # Create checkpoint manager.
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(embedder), checkpoint_dir, device)

    global_step = checkpoint_manager.restore_or_initialize()
    print(f"Restored model from checkpoint {global_step}")

    return embedder, loaders


def main(args):
    embedder, loaders = setup_embedder(
        'comb_gripper',
        '/home/kevin/repos/kronos/configs/magical_config/gripper.yml')
    device = torch.device('cuda')
    embedder.to(device)
    goal_emb = compute_goal_embedding(embedder, loaders, device)
    register_envs()
    env = gym.make(ENV_NAME.format(args.agent.capitalize()))
    viewer = KeyboardEnvInteractor()
    run_episode(env, viewer, embedder, goal_emb, device, args.debug)


def run_episode(env, viewer, embedder, goal_emb, device, debug):
    seed = np.random.randint(0, 100)
    env.seed(seed)
    obs = env.reset()
    record_episode(seed, env, viewer, obs, embedder, goal_emb, device, debug)


def plot_reward(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.show()
    plt.close()


def record_episode(seed, env, viewer, obs, embedder, goal_emb, device, debug):
    # Save the first observation.
    obs = env.render('rgb_array')
    viewer.imshow(obs)

    i = [1]
    rewards = []

    def step(action):
        _, rew, done, info = env.step(action)
        rew = compute_reward(
            env.render('rgb_array'), goal_emb, embedder, device)
        rewards.append(rew)
        obs = env.render('rgb_array')
        if i[0] % 50 == 0:
            print(f"Done, score {info['eval_score']:.4g}/1.0")
        i[0] += 1
        return obs

    viewer.run_loop(step)

    plot_reward(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default='gripper')
    parser.add_argument(
        "--debug", type=lambda s: s.lower() in ["true", "1"], default=False
    )
    args = parser.parse_args()
    main(args)
