import matplotlib.pyplot as plt
from typing import Tuple, List
import csv
import os, sys
import numpy as np


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


def plot(progress_data, fig, ax, checkpoint_dir, plot_script=False):
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

    def smoothed(data, window_sz=100):
        if len(data) < window_sz:
            return data
        return np.convolve(data, np.ones(window_sz) / window_sz, mode="valid")

    scores = smoothed(scores)
    best_scores = smoothed(best_scores)
    episodes = episodes[: len(scores)]

    # scores
    ax[0].plot(
        episodes,
        scores,
        label="Score",
        marker="o",
        markersize=1,
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
        markersize=1,
        linestyle="--",
        color="r",
    )
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Best Score")
    ax[1].legend()
    ax[1].grid(True)

    if plot_script:
        fig.savefig("plot.png")
    else:
        fig.savefig(os.path.join(checkpoint_dir, "scores.png"))
