"""Visualize LunarLander FQI or DQN results.

Usage:
    python visualize_results.py fqi
    python visualize_results.py dqn
"""

import argparse
from dataclasses import dataclass
import pathlib
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BASE_DIR = pathlib.Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"

plt.rcParams.update({"text.usetex": False, "font.family": "serif"})


def flatten_tasks(data):
    """Flatten (seed, task, iteration) into (seed, global iteration)."""
    return data.reshape(data.shape[0], -1)


def add_mean_if_aligned(ax, curves):
    """Plot a mean only when all curves share the exact same x axis."""
    if len(curves) < 2:
        return
    reference_x = curves[0][0]
    if not all(np.array_equal(x, reference_x) for x, _ in curves[1:]):
        warnings.warn("Different x axes: the mean curve was not computed")
        return
    values = np.stack([values for _, values in curves])
    ax.plot(reference_x, values.mean(axis=0), color="black", linewidth=2.5,
            label=f"Mean ({len(curves)} seeds)")


def finish_figure(figure, ax, output_path, xlabel, ylabel, title,
                  legend_columns=2):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=legend_columns)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


@dataclass(frozen=True)
class FQIConfiguration:
    folder: str
    color: str
    label: str
    linestyle: str


class FQIVisualizer:
    FOLDER_PATTERN = re.compile(
        r"(?P<learner>.+)_"
        r"(?P<boosting>no_boosted|boosted)_"
        r"(?P<curriculum>no_curriculum|curriculum)"
    )
    CONDITIONS = {
        ("boosted", "curriculum"): ("C0", "BC-FQI"),
        ("boosted", "no_curriculum"): ("C1", "B-FQI"),
        ("no_boosted", "curriculum"): ("C2", "C-FQI"),
        ("no_boosted", "no_curriculum"): ("C3", "FQI"),
    }
    def __init__(self):
        self.logs_dir = LOGS_DIR
        self.output_dir = FIGURES_DIR / "fqi"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learners = self._discover_configurations()
        self.configurations = []
        self.current_output_dir = self.output_dir

    def _discover_configurations(self):
        learners = {}
        for folder in sorted(self.logs_dir.iterdir() if self.logs_dir.exists()
                             else []):
            if not folder.is_dir():
                continue
            match = self.FOLDER_PATTERN.fullmatch(folder.name)
            if match is None:
                continue
            learner = match.group("learner")
            condition = (
                match.group("boosting"), match.group("curriculum")
            )
            learners.setdefault(learner, {})[condition] = folder.name

        if not learners:
            raise FileNotFoundError(
                f"No FQI result directory matching "
                f"<learner>_<boosting>_<curriculum> in {self.logs_dir}"
            )

        discovered = {}
        for learner, folders in sorted(learners.items()):
            missing = set(self.CONDITIONS) - set(folders)
            if missing:
                warnings.warn(
                    f"FQI learner '{learner}' is missing configurations: "
                    f"{sorted(missing)}"
                )
            configurations = []
            for condition, folder_name in folders.items():
                color, base_label = self.CONDITIONS[condition]
                configurations.append(FQIConfiguration(
                    folder=folder_name,
                    color=color,
                    label=base_label,
                    linestyle="-",
                ))
            discovered[learner] = configurations
        return discovered

    def _load_metric(self, filename):
        loaded = []
        for configuration in self.configurations:
            path = self.logs_dir / configuration.folder / filename
            if not path.exists():
                warnings.warn(
                    f"{filename}: missing configuration "
                    f"{configuration.label} ({path})"
                )
                continue
            loaded.append((configuration, np.load(path)))

        if not loaded:
            warnings.warn(f"No {filename} file found; plot skipped")
            return []

        seed_counts = {
            configuration.label: len(data)
            for configuration, data in loaded
        }
        expected = max(seed_counts.values())
        incomplete = {
            label: count for label, count in seed_counts.items()
            if count != expected
        }
        if incomplete:
            warnings.warn(
                f"{filename} does not contain all seeds: {seed_counts}"
            )
        return loaded

    @staticmethod
    def _add_task_regions(ax, n_tasks, iterations_per_task):
        for task_idx in range(n_tasks):
            start = task_idx * iterations_per_task
            end = (task_idx + 1) * iterations_per_task
            if task_idx % 2 == 0:
                ax.axvspan(start, end, color="black", alpha=0.06, zorder=0)
            if task_idx > 0:
                ax.axvline(start, color="black", linestyle="--",
                           linewidth=1, alpha=0.5)
            ax.text(
                (start + end) / 2, 1.01,
                rf"$\mathcal{{T}}_{task_idx + 1}$",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            )

    @staticmethod
    def _plot_seed_curves(ax, data, configuration):
        x = np.arange(1, data.shape[1] + 1)
        ax.plot(x, data.T, color=configuration.color, alpha=0.22,
                linestyle=configuration.linestyle, linewidth=0.7)
        ax.plot(x, data.mean(axis=0), color=configuration.color,
                linestyle=configuration.linestyle, linewidth=2,
                label=configuration.label)

    def plot_returns(self, final_task_only=False):
        loaded = self._load_metric("J.npy")
        if not loaded:
            return
        figure, ax = plt.subplots(figsize=(8, 4.5))
        valid_reference = None

        for configuration, returns in loaded:
            if returns.ndim != 3:
                warnings.warn(
                    f"J.npy for {configuration.label} has invalid shape "
                    f"{returns.shape}; expected (seeds, tasks, iterations)"
                )
                continue
            valid_reference = returns
            values = returns[:, -1, :] if final_task_only \
                else flatten_tasks(returns)
            self._plot_seed_curves(ax, values, configuration)

        if not ax.lines:
            plt.close(figure)
            return
        if not final_task_only:
            self._add_task_regions(
                ax, valid_reference.shape[1], valid_reference.shape[2]
            )
            output = self.current_output_dir / "lunarlander_performance.pdf"
            xlabel = "FQI iteration"
            title = "LunarLander performance across tasks"
        else:
            output = self.current_output_dir / "lunarlander_target_performance.pdf"
            xlabel = "FQI iteration on final task"
            title = "LunarLander final-task performance"

        finish_figure(
            figure, ax, output, xlabel, "Cumulative discounted return", title
        )

    def plot_depths(self):
        loaded = self._load_metric("depths.npy")
        if not loaded:
            return
        figure, ax = plt.subplots(figsize=(8, 4.5))
        valid_reference = None

        for configuration, depths in loaded:
            if depths.ndim != 4:
                warnings.warn(
                    f"depths.npy for {configuration.label} has invalid shape "
                    f"{depths.shape}; expected (seeds, tasks, iterations, trees)"
                )
                continue
            valid_reference = depths
            seed_means = flatten_tasks(depths.mean(axis=-1))
            tree_std = flatten_tasks(depths.std(axis=-1)).mean(axis=0)
            x = np.arange(1, seed_means.shape[1] + 1)
            mean = seed_means.mean(axis=0)
            ax.fill_between(
                x, mean - tree_std, mean + tree_std,
                color=configuration.color, alpha=0.12,
            )
            ax.plot(x, seed_means.T, color=configuration.color, alpha=0.20,
                    linestyle=configuration.linestyle, linewidth=0.7)
            ax.plot(x, mean, color=configuration.color, linewidth=2,
                    linestyle=configuration.linestyle,
                    label=configuration.label)

        if valid_reference is None:
            plt.close(figure)
            return
        self._add_task_regions(
            ax, valid_reference.shape[1], valid_reference.shape[2]
        )
        finish_figure(
            figure, ax, self.current_output_dir / "lunarlander_depths.pdf",
            "FQI iteration", "Mean tree depth",
            "Evolution of ExtraTrees depth",
        )

    def run(self):
        for learner, configurations in self.learners.items():
            self.configurations = configurations
            self.current_output_dir = self.output_dir / learner
            self.current_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Visualizing FQI learner: {learner}")
            self.plot_returns(final_task_only=False)
            self.plot_returns(final_task_only=True)
            self.plot_depths()


@dataclass(frozen=True)
class DQNMetric:
    raw: str
    smooth: str
    x: str
    filename: str
    xlabel: str
    ylabel: str
    title: str


class DQNFileIndex:
    """Index per-seed NumPy files, accepting '-' and '_' in metric names."""

    FILE_PATTERN = re.compile(r"(.+)-(\d+)\.npy")

    def __init__(self, directory):
        self.directory = directory
        self.files = {}
        for path in directory.glob("*.npy"):
            match = self.FILE_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            metric, seed = match.groups()
            canonical_metric = metric.replace("-", "_")
            self.files.setdefault(canonical_metric, {})[int(seed)] = path

        self.all_seeds = sorted({
            seed for metric_files in self.files.values()
            for seed in metric_files
        })
        if not self.all_seeds:
            raise FileNotFoundError(
                f"No per-seed DQN metric found in {directory}"
            )

    def seeds(self, metric):
        return set(self.files.get(metric, {}))

    def load(self, metric, seed):
        return np.load(self.files[metric][seed])

    def complete_seeds(self, metrics, display_name):
        available_sets = [self.seeds(metric) for metric in metrics]
        complete = set.intersection(*available_sets) if available_sets else set()
        expected = set(self.all_seeds)
        if complete != expected:
            missing = sorted(expected - complete)
            details = {
                metric: sorted(self.seeds(metric)) for metric in metrics
            }
            warnings.warn(
                f"{display_name} does not have all seeds. "
                f"Complete seeds: {sorted(complete)}; missing: {missing}; "
                f"files by metric: {details}"
            )
        return sorted(complete)


class DQNVisualizer:
    METRICS = (
        DQNMetric(
            "training_rewards_raw", "training_rewards",
            "training_reward_steps", "dqn_training_rewards.pdf",
            "Environment steps", "Immediate reward",
            "DQN training reward (rolling mean: 500 steps)",
        ),
        DQNMetric(
            "episode_returns_raw", "episode_returns", "episode_indices",
            "dqn_episode_returns.pdf", "Episode", "Episode return",
            "DQN episode return (rolling mean: 50 episodes)",
        ),
        DQNMetric(
            "episode_lengths_raw", "episode_lengths", "episode_indices",
            "dqn_episode_lengths.pdf", "Episode",
            "Episode length (steps)",
            "DQN episode length (rolling mean: 50 episodes)",
        ),
        DQNMetric(
            "losses_raw", "losses", "loss_steps", "dqn_td_loss.pdf",
            "Environment steps", "TD loss (MSE)",
            "DQN TD loss (rolling mean: 50 training points)",
        ),
    )

    def __init__(self):
        self.logs_dir = LOGS_DIR / "dqn_lunarlander"
        self.output_dir = FIGURES_DIR / "dqn"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = DQNFileIndex(self.logs_dir)

    def plot_metric(self, metric):
        seeds = self.index.complete_seeds(
            (metric.raw, metric.smooth, metric.x), metric.ylabel
        )
        if not seeds:
            warnings.warn(f"No complete data for {metric.ylabel}; plot skipped")
            return

        figure, ax = plt.subplots(figsize=(7, 4.5))
        colors = plt.get_cmap("tab10")
        smooth_curves = []

        for color_idx, seed in enumerate(seeds):
            x = self.index.load(metric.x, seed)
            raw = self.index.load(metric.raw, seed)
            smooth = self.index.load(metric.smooth, seed)
            if not (len(x) == len(raw) == len(smooth)):
                warnings.warn(
                    f"{metric.ylabel}: inconsistent lengths for seed {seed}; "
                    "seed skipped"
                )
                continue
            color = colors(color_idx % 10)
            ax.plot(x, raw, color=color, alpha=0.12, linewidth=0.5)
            ax.plot(x, smooth, color=color, linewidth=1.5,
                    label=f"Seed {seed}")
            smooth_curves.append((x, smooth))

        if not smooth_curves:
            plt.close(figure)
            return
        add_mean_if_aligned(ax, smooth_curves)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="gray", alpha=0.25))
        labels.append("Raw values")
        ax.legend(handles, labels, ncol=2)
        ax.set(xlabel=metric.xlabel, ylabel=metric.ylabel,
               title=metric.title)
        ax.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.output_dir / metric.filename,
                       bbox_inches="tight")
        plt.close(figure)

    def plot_evaluation(self):
        seeds = self.index.complete_seeds(("J",), "Evaluation return")
        if not seeds:
            warnings.warn("No evaluation returns; plot skipped")
            return
        figure, ax = plt.subplots(figsize=(7, 4.5))
        colors = plt.get_cmap("tab10")
        curves = []
        use_environment_steps = all(
            seed in self.index.seeds("training_reward_steps")
            for seed in seeds
        )
        if not use_environment_steps:
            warnings.warn(
                "Some evaluation seeds have no training step axis; "
                "using evaluation checkpoints for every seed"
            )

        for color_idx, seed in enumerate(seeds):
            returns = self.index.load("J", seed)
            if use_environment_steps:
                training_steps = self.index.load(
                    "training_reward_steps", seed
                )
                total_steps = training_steps[-1]
                x = np.linspace(total_steps / len(returns), total_steps,
                                len(returns))
                xlabel = "Environment steps"
            else:
                x = np.arange(1, len(returns) + 1)
                xlabel = "Evaluation checkpoint"
            color = colors(color_idx % 10)
            ax.plot(x, returns, color=color, linewidth=1.5, marker="o",
                    markersize=3, label=f"Seed {seed}")
            curves.append((x, returns))

        add_mean_if_aligned(ax, curves)
        finish_figure(
            figure, ax, self.output_dir / "dqn_evaluation_returns.pdf",
            xlabel, "Cumulative discounted return",
            "DQN evaluation performance",
        )

    def run(self):
        print(f"Detected DQN seeds: {self.index.all_seeds}")
        self.plot_evaluation()
        for metric in self.METRICS:
            self.plot_metric(metric)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "algorithm", choices=("fqi", "dqn"),
        help="Type of experiment to visualize",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualizer = FQIVisualizer() if args.algorithm == "fqi" \
        else DQNVisualizer()
    visualizer.run()
    print(f"Saved figures to {visualizer.output_dir}")
