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


def load_progress(checkpoint_dir) -> Tuple[List, bool]:
    """
    loads progress from csv, looks in checkpoint_dir
    columns: episode(int), score, best_score
    """
    file_path = os.path.join(checkpoint_dir, "progress.csv")
    progress_data = []

    if not os.path.exists(file_path):
        print(f"No progress file found at {file_path}. Starting fresh.")
        return progress_data, False

    with open(file_path, mode="r") as file:
        for row in csv.DictReader(file):
            row["episode"] = int(row["episode"])
            row["score"] = float(row["score"])
            row["best_score"] = float(row["best_score"])
            progress_data.append(row)

    ckpt_found = os.path.exists(os.path.join(checkpoint_dir, "actor.pth"))
    return progress_data, ckpt_found


def save_progress(checkpoint_dir, episode, score, best_score) -> None:
    """
    save progress to csv, looks in checkpoint_dir
    columns: episode(int), score, best_score
    """
    file_path = os.path.join(checkpoint_dir, "progress.csv")
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="") as file:
        fieldnames = ["episode", "score", "best_score"]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            csv_writer.writeheader()

        csv_writer.writerow(
            {"episode": episode, "score": score, "best_score": best_score}
        )


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
    matplotlib.use("Agg")
    args = parse_args()

    # config
    mode = args.mode
    env_config = {
        "name": "InvertedDoublePendulum-v5",
        "slug": "invp_v5",
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
            obs_, reward, done, truncated, info = env.step(action)
            score += float(reward)
            agent.remember(obs, action, reward, obs_, done or truncated)
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
            episodes = []
            scores = []
            best_scores = []
            row: dict
            for row in progress_data:
                e, s, bs = list(row.values())
                episodes.append(e)
                scores.append(s)
                best_scores.append(bs)

            ax[0].clear()
            ax[1].clear()

            # scores
            ax[0].plot(
                episodes,
                scores,
                label="Score",
                marker="o",
                markersize=2,
                linestyle="-",
                color="b",
            )
            ax[0].set_xlabel("Episode")
            ax[0].set_ylabel("Score")
            ax[0].legend()
            ax[0].grid(True)

            # best scores
            ax[1].plot(
                episodes,
                best_scores,
                label="Best Score",
                marker="x",
                markersize=2,
                linestyle="--",
                color="r",
            )
            ax[1].set_xlabel("Episode")
            ax[1].set_ylabel("Best Score")
            ax[1].legend()
            ax[1].grid(True)

            fig.savefig(os.path.join(checkpoint_dir, "scores.png"))
