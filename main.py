from math import inf
import time
import gymnasium as gym
import numpy as np
import csv
import os
from agent import Agent
from typing import Literal, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import argparse
from utils import plot, save_progress, load_progress
import pygame


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        choices=["replay", "train"],
        help="Script op mode",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=256,
        help="Training batch size",
    )

    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=50_000,
        help="Number of training episodes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    screen = pygame.display.set_mode((400, 300))
    matplotlib.use("Agg")
    args = parse_args()

    # config
    mode = args.mode
    env_config = {
        "name": "InvertedDoublePendulum-v4",
        "slug": "invp_v4",
    }
    train_config = {
        "batch_size": args.batch_size,
        "n_eps": args.episodes,
    }
    checkpoint_dir = f"checkpoints/{env_config['slug']}"
    if mode == "replay":
        env = gym.make(
            env_config["name"],
            render_mode="human",
        )
    else:
        env = gym.make(env_config["name"])

    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    batch_size = train_config["batch_size"]
    n_eps = train_config["n_eps"]

    agent = Agent(
        n_obs,
        n_actions,
        max_action,
        alpha=0.2,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    progress_data, load_checkpoint = load_progress(checkpoint_dir)
    best_score = -inf if len(progress_data) == 0 else progress_data[-1]["best_score"]
    ep = 0 if len(progress_data) == 0 else progress_data[-1]["episode"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    if load_checkpoint:
        print("Found checkpoint. Loading...")
        agent.load_models()
    else:
        print("No checkpoint found.")

    if mode == "replay":
        n_eps = 100
        ep = 0
    for _ in range(n_eps - ep):
        ep += 1
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(obs)
            if mode == "replay":
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                        action = np.array([-max_action * 0.7])
                        print("Manual override: Moving Left")
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                        action = np.array([max_action * 0.7])
                        print("Manual override: Moving Right")

            obs_, reward, done, truncated, info = env.step(action)
            score += float(reward)
            agent.remember(obs, action, reward, obs_, done or truncated)
            done = done or truncated
            if mode != "replay":
                agent.learn()
            obs = obs_

        if mode != "replay":
            if score > best_score:
                best_score = score
                agent.save_models()
                print(
                    f"-->[*] Episode {ep} \t\t Score: {score:.2f} \t\t Best Score: {best_score:.2f}"
                )
            else:
                print(
                    f"Episode {ep} \t\t Score: {score:.2f} \t\t Best Score: {best_score:.2f}"
                )

            save_progress(checkpoint_dir, ep, score, best_score)
            progress_data, _ = load_progress(checkpoint_dir)

        if ep % 10 == 0:
            # make progress plot
            plot(progress_data, fig, ax, checkpoint_dir)
