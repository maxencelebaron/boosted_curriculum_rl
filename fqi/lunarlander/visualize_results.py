"""Visualize tree-FQI results produced by ``run_tree_jobs.sh``."""

import argparse
import pathlib
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


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


def find_metric_file(log_dir, metric, seed):
    """Support both metric_name-95.npy and metric-name-95.npy."""
    candidates = (
        log_dir / f"{metric}-{seed}.npy",
        log_dir / f"{metric.replace('_', '-')}-{seed}.npy",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_dqn_seeds(log_dir):
    seeds = []
    for path in log_dir.glob("J-*.npy"):
        match = re.fullmatch(r"J-(\d+)\.npy", path.name)
        if match is not None:
            seeds.append(int(match.group(1)))
    if not seeds:
        raise FileNotFoundError(f"No J-<seed>.npy found in {log_dir}")
    return sorted(set(seeds))


def load_dqn_metric(log_dir, seeds, metric):
    data = []
    for seed in seeds:
        path = find_metric_file(log_dir, metric, seed)
        if path is None:
            warnings.warn(f"Missing {metric} for seed {seed}")
            continue
        data.append((seed, np.load(path)))
    return data


def add_mean_if_aligned(ax, curves, color="black"):
    if len(curves) < 2:
        return
    reference_x = curves[0][1]
    if not all(np.array_equal(x, reference_x) for _, x, _ in curves[1:]):
        warnings.warn("Curves use different x axes; mean curve was skipped")
        return
    values = np.stack([y for _, _, y in curves])
    ax.plot(reference_x, values.mean(axis=0), color=color, linewidth=2.5,
            label=f"Mean ({len(curves)} seeds)")


def plot_dqn_metric(log_dir, seeds, output_path, raw_metric,
                    smoothed_metric, x_metric, xlabel, ylabel, title):
    figure, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.get_cmap("tab10")
    plotted_curves = []

    for color_idx, seed in enumerate(seeds):
        x_path = find_metric_file(log_dir, x_metric, seed)
        raw_path = find_metric_file(log_dir, raw_metric, seed)
        smooth_path = find_metric_file(log_dir, smoothed_metric, seed)
        if x_path is None or raw_path is None or smooth_path is None:
            warnings.warn(f"Skipping incomplete metric files for seed {seed}")
            continue

        x = np.load(x_path)
        raw = np.load(raw_path)
        smooth = np.load(smooth_path)
        if not (len(x) == len(raw) == len(smooth)):
            warnings.warn(f"Skipping seed {seed}: inconsistent array lengths")
            continue

        color = colors(color_idx % 10)
        ax.plot(x, raw, color=color, alpha=0.12, linewidth=0.5)
        ax.plot(x, smooth, color=color, linewidth=1.5,
                label=f"Seed {seed}")
        plotted_curves.append((seed, x, smooth))

    if not plotted_curves:
        plt.close(figure)
        warnings.warn(f"No complete data available for {title}")
        return

    add_mean_if_aligned(ax, plotted_curves)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", alpha=0.25,
                          linewidth=1))
    labels.append("Raw values")
    ax.legend(handles, labels, ncol=2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def visualize_dqn_evaluation(log_dir, seeds, output_path):
    figure, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.get_cmap("tab10")
    plotted_curves = []

    for color_idx, seed in enumerate(seeds):
        returns = np.load(log_dir / f"J-{seed}.npy")
        reward_steps_path = find_metric_file(
            log_dir, "training_reward_steps", seed
        )
        if reward_steps_path is not None:
            total_steps = np.load(reward_steps_path)[-1]
            x = np.linspace(total_steps / len(returns), total_steps,
                            len(returns))
            xlabel = "Environment steps"
        else:
            x = np.arange(1, len(returns) + 1)
            xlabel = "Evaluation checkpoint"

        color = colors(color_idx % 10)
        ax.plot(x, returns, color=color, linewidth=1.5,
                marker="o", markersize=3, label=f"Seed {seed}")
        plotted_curves.append((seed, x, returns))

    add_mean_if_aligned(ax, plotted_curves)
    ax.set(xlabel=xlabel, ylabel="Cumulative discounted return",
           title="DQN evaluation performance")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def visualize_dqn_results(log_dir, output_dir):
    seeds = discover_dqn_seeds(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found completed seeds: {seeds}")

    visualize_dqn_evaluation(
        log_dir, seeds, output_dir / "dqn_evaluation_returns.pdf"
    )
    plot_dqn_metric(
        log_dir, seeds, output_dir / "dqn_training_rewards.pdf",
        "training_rewards_raw", "training_rewards",
        "training_reward_steps", "Environment steps", "Immediate reward",
        "DQN training reward (rolling mean: 500 steps)",
    )
    plot_dqn_metric(
        log_dir, seeds, output_dir / "dqn_episode_returns.pdf",
        "episode_returns_raw", "episode_returns", "episode_indices",
        "Episode", "Episode return",
        "DQN training episode return (rolling mean: 50 episodes)",
    )
    plot_dqn_metric(
        log_dir, seeds, output_dir / "dqn_episode_lengths.pdf",
        "episode_lengths_raw", "episode_lengths", "episode_indices",
        "Episode", "Episode length (steps)",
        "DQN training episode length (rolling mean: 50 episodes)",
    )
    plot_dqn_metric(
        log_dir, seeds, output_dir / "dqn_td_loss.pdf",
        "losses_raw", "losses", "loss_steps",
        "Environment steps", "TD loss (MSE)",
        "DQN TD loss (rolling mean: 50 training points)",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=pathlib.Path, default=pathlib.Path("logs"))
    parser.add_argument(
        "--dqn-dir", type=pathlib.Path, default=None,
        help="DQN result directory; when provided, plot DQN instead of Tree-FQI",
    )
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dqn_dir is not None:
        visualize_dqn_results(args.dqn_dir, args.output_dir)
    else:
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
