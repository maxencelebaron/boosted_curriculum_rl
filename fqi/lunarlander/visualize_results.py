"""Visualize tree-FQI results produced by ``run_tree_jobs.sh``."""

import argparse
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


CONFIGURATIONS = (
    ("trees_boosted_curriculum", "C0", "BC-FQI"),
    ("trees_boosted_no_curriculum", "C1", "B-FQI"),
    ("trees_no_boosted_curriculum", "C2", "C-FQI"),
    ("trees_no_boosted_no_curriculum", "C3", "FQI"),
)

plt.rcParams.update({"text.usetex": False, "font.family": "serif"})


def load_results(logs_dir):
    results = []
    for folder_name, color, label in CONFIGURATIONS:
        folder = logs_dir / folder_name
        try:
            returns = np.load(folder / "J.npy")
            depths = np.load(folder / "depths.npy")
            with (folder / "config.yaml").open("r") as stream:
                config = yaml.safe_load(stream)
        except FileNotFoundError as error:
            warnings.warn(f"Skipping {label}: {error}")
            continue

        if returns.ndim != 3:
            raise ValueError(
                f"{folder / 'J.npy'} must have shape "
                "(n_exp, n_tasks, n_iterations)"
            )
        if depths.ndim != 4:
            raise ValueError(
                f"{folder / 'depths.npy'} must have shape "
                "(n_exp, n_tasks, n_iterations, n_trees)"
            )
        results.append({
            "label": label,
            "color": color,
            "returns": returns,
            "depths": depths,
            "config": config,
        })

    if not results:
        raise FileNotFoundError(f"No tree-FQI results found in {logs_dir}")
    return results


def flatten_tasks(data):
    """Convert (experiments, tasks, iterations) to one training timeline."""
    return data.reshape(data.shape[0], -1)


def add_task_regions(ax, n_tasks, iterations_per_task):
    for task_idx in range(n_tasks):
        start = task_idx * iterations_per_task
        end = (task_idx + 1) * iterations_per_task
        if task_idx % 2 == 0:
            ax.axvspan(start, end, color="black", alpha=0.06, zorder=0)
        if task_idx > 0:
            ax.axvline(start, color="black", linestyle="--",
                       linewidth=1, alpha=0.5)
        ax.text(
            (start + end) / 2,
            1.01,
            rf"$\mathcal{{T}}_{task_idx + 1}$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
        )


def plot_seed_curves(ax, data, color, label):
    x = np.arange(1, data.shape[1] + 1)
    ax.plot(x, data.T, color=color, alpha=0.22,
            linestyle="--", linewidth=0.7)
    mean = data.mean(axis=0)
    line, = ax.plot(x, mean, color=color, linewidth=2, label=label)
    return line


def visualize_returns(results, output_path):
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for result in results:
        plot_seed_curves(
            ax,
            flatten_tasks(result["returns"]),
            result["color"],
            result["label"],
        )

    reference = results[0]["returns"]
    add_task_regions(ax, reference.shape[1], reference.shape[2])
    ax.set(xlabel="FQI iteration", ylabel="Cumulative discounted return",
           title="LunarLander performance across tasks")
    ax.set_xlim(0, reference.shape[1] * reference.shape[2])
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def visualize_target_returns(results, output_path):
    figure, ax = plt.subplots(figsize=(6, 4))
    for result in results:
        plot_seed_curves(
            ax,
            result["returns"][:, -1, :],
            result["color"],
            result["label"],
        )

    ax.set(xlabel="FQI iteration on final task",
           ylabel="Cumulative discounted return",
           title="LunarLander final-task performance")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def visualize_depths(results, output_path):
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for result in results:
        depths = result["depths"]
        mean_per_experiment = depths.mean(axis=-1)
        tree_std = depths.std(axis=-1)
        timeline = flatten_tasks(mean_per_experiment)
        std_timeline = flatten_tasks(tree_std).mean(axis=0)
        x = np.arange(1, timeline.shape[1] + 1)
        mean = timeline.mean(axis=0)

        ax.fill_between(
            x, mean - std_timeline, mean + std_timeline,
            color=result["color"], alpha=0.12,
        )
        ax.plot(x, timeline.T, color=result["color"], alpha=0.20,
                linestyle="--", linewidth=0.7)
        ax.plot(x, mean, color=result["color"], linewidth=2,
                label=result["label"])

    reference = results[0]["depths"]
    add_task_regions(ax, reference.shape[1], reference.shape[2])
    ax.set(xlabel="FQI iteration", ylabel="Mean tree depth",
           title="Evolution of ExtraTrees depth")
    ax.set_xlim(0, reference.shape[1] * reference.shape[2])
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=pathlib.Path, default=pathlib.Path("logs"))
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded_results = load_results(args.logs_dir)
    visualize_returns(
        loaded_results, args.output_dir / "lunarlander_performance.pdf"
    )
    visualize_target_returns(
        loaded_results,
        args.output_dir / "lunarlander_target_performance.pdf",
    )
    visualize_depths(
        loaded_results, args.output_dir / "lunarlander_depths.pdf"
    )
    print(f"Saved figures to {args.output_dir}")
