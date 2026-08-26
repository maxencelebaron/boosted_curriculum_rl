"""Visualize LunarLander FQI or DQN results.

Usage:
    python visualize_results.py fqi
    python visualize_results.py dqn
"""

import argparse
from dataclasses import dataclass
import pathlib
import re
from typing import Optional
import warnings
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


BASE_DIR = pathlib.Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"
MEAN_COLOR = "C3"
AXES_BACKGROUND = "#E2E2E2"

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": AXES_BACKGROUND,
    "axes.edgecolor": "#AEB4BA",
    "grid.color": "white",
    "grid.linewidth": 0.8,
})


def flatten_tasks(data):
    """Flatten (seed, task, iteration) into (seed, global iteration)."""
    return data.reshape(data.shape[0], -1)


def moving_average(values, window):
    """Return a trailing moving average with a shorter initial window."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    cumulative = np.cumsum(np.r_[0.0, values])
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - window + 1)
    return (cumulative[indices + 1] - cumulative[starts]) / (
        indices - starts + 1
    )


def add_mean_if_aligned(ax, curves):
    """Plot a mean only when all curves share the exact same x axis."""
    if len(curves) < 2:
        return
    reference_x = curves[0][0]
    if not all(np.array_equal(x, reference_x) for x, _ in curves[1:]):
        warnings.warn("Different x axes: the mean curve was not computed")
        return
    values = np.stack([values for _, values in curves])
    ax.plot(reference_x, values.mean(axis=0), color=MEAN_COLOR, linewidth=1.5,
            label=f"Mean ({len(curves)} seeds)")


def add_episode_mean(ax, curves):
    """Average every seed over their common episode-index prefix."""
    if len(curves) < 2:
        return
    common_length = min(len(x) for x, _ in curves)
    reference_x = curves[0][0][:common_length]
    if not all(
        np.array_equal(x[:common_length], reference_x)
        for x, _ in curves[1:]
    ):
        warnings.warn("Episode axes are inconsistent; mean was not computed")
        return
    values = np.stack([y[:common_length] for _, y in curves])
    ax.plot(reference_x, values.mean(axis=0), color=MEAN_COLOR, linewidth=1.5,
            label=f"Mean ({len(curves)} seeds)")


def finish_figure(figure, ax, output_path, xlabel, ylabel, title, legend_columns=2, title_pad=6):
    ax.set_facecolor(AXES_BACKGROUND)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, pad=title_pad)
    ax.grid(alpha=0.85)
    ax.legend(
        ncol=legend_columns,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
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
        # Keep the learner match non-greedy so that, for example,
        # ``neural_no_boosted_curriculum`` is split at ``no_boosted``
        # rather than being interpreted as learner ``neural_no``.
        r"(?P<learner>.+?)_"
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
        for folder in sorted(self.logs_dir.iterdir() if self.logs_dir.exists() else []):
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
            figure, ax, output, xlabel, "Cumulative discounted return", title,
            title_pad=28 if not final_task_only else 6,
        )

    def plot_depths(self):
        # Neural FQI runs do not produce tree-depth metrics (see run.py).
        # Skip this plot quietly when none of this learner's configurations
        # contains the metric, while retaining per-configuration warnings for
        # genuinely incomplete ExtraTrees results.
        if not any(
            (self.logs_dir / configuration.folder / "depths.npy").exists()
            for configuration in self.configurations
        ):
            return
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
            title_pad=28,
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
    values: str
    x: str
    filename: str
    xlabel: str
    ylabel: str
    title: str
    mean_by_episode: bool = False
    legacy_values: Optional[str] = None
    smoothing_window: Optional[int] = None


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
        return np.load(self.files[metric][seed], allow_pickle=False)


@dataclass
class DQNExperiment:
    directory: pathlib.Path
    key: str
    label: str
    color: object
    is_growth: bool
    index: DQNFileIndex
    configs: dict

    @property
    def seeds(self):
        return self.index.all_seeds


class DQNVisualizer:
    BASELINE_FOLDER = "dqn_lunarlander"
    GROWTH_PREFIX = "dqn_lunarlander_grow_"
    KNOWN_COLORS = {
        "baseline": "C3",
        "random": "C0",
        "random-0": "C1",
        "svd": "C2",
        "gromo_one_layer": "C4",
    }
    DISPLAY_LABELS = {
        "baseline": "Baseline",
        "gromo_one_layer": "Tiny",
        "svd": "ALS",
    }

    UNSMOOTHED_METRICS = (
        DQNMetric(
            "feature_rank",
            "monitoring_steps",
            "dqn_feature_rank.pdf",
            "Environment steps",
            "Feature rank",
            "DQN feature rank",
        ),
        DQNMetric(
            "feature_rank_ratio",
            "monitoring_steps",
            "dqn_feature_rank_ratio.pdf",
            "Environment steps",
            "Feature rank / feature dimension",
            "DQN normalized feature rank",
        ),
        DQNMetric(
            "feature_srank",
            "monitoring_steps",
            "dqn_feature_srank.pdf",
            "Environment steps",
            "Stable rank",
            "DQN feature stable rank",
        ),
        DQNMetric(
            "feature_srank_ratio", "monitoring_steps",
            "dqn_feature_srank_ratio.pdf", "Environment steps",
            "Stable rank / feature dimension",
            "DQN normalized feature stable rank",
        ),
        DQNMetric(
            "plasticity", "plasticity_steps", "dqn_plasticity.pdf",
            "Environment steps", "Plasticity",
            "DQN plasticity",
        ),
    )

    # Files produced for every seed by the current training scripts. Only raw
    # training series are stored; smoothing belongs to the visualizer.
    COMMON_FILES = {
        "J", "training_reward_steps",
        "training_rewards_raw", "episode_indices", "episode_returns_raw",
        "episode_lengths_raw", "loss_steps", "losses_raw",
        "monitoring_steps", "monitoring_task_indices", "feature_rank",
        "feature_rank_ratio", "feature_srank", "feature_srank_ratio",
    }
    TASK_METADATA_FILES = {
        "evaluation_steps", "evaluation_task_indices", "task_boundaries",
        "task_wind_powers", "task_timesteps", "task_evaluations",
    }
    PLASTICITY_FILES = {
        "plasticity_steps", "plasticity_task_indices", "plasticity",
    }
    GENERATION_METRICS = {
        "principal_angle_min": "Minimum principal-angle cosine",
        "principal_angle_max": "Maximum principal-angle cosine",
        "principal_angle_mean": "Mean principal-angle cosine",
        "brc_normalized": "Normalized Bellman-residual alignment",
    }
    GENERATION_FILES = {
        "bounds", "growth_step", "steps", *GENERATION_METRICS,
    }

    LEGACY_METRICS = {
        "training_rewards_raw": "training_rewards",
        "episode_returns_raw": "episode_returns",
        "episode_lengths_raw": "episode_lengths",
        "losses_raw": "losses",
    }

    def __init__(self, reward_window=500, episode_window=50, loss_window=50):
        for name, value in {
            "reward_window": reward_window,
            "episode_window": episode_window,
            "loss_window": loss_window,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.metrics = (
            DQNMetric(
                "J", "evaluation_steps", "dqn_evaluation_returns.pdf",
                "Environment steps", "Cumulative discounted return",
                "DQN evaluation performance",
            ),
            DQNMetric(
                "training_rewards_raw", "training_reward_steps",
                "dqn_training_rewards.pdf", "Environment steps",
                "Immediate reward",
                "Smoothed DQN training reward "
                f"({reward_window}-step moving average)",
                legacy_values="training_rewards",
                smoothing_window=reward_window,
            ),
            DQNMetric(
                "episode_returns_raw", "episode_indices",
                "dqn_episode_returns.pdf", "Episode", "Episode return",
                "Smoothed DQN episodic return "
                f"({episode_window}-episode moving average)",
                mean_by_episode=True,
                legacy_values="episode_returns",
                smoothing_window=episode_window,
            ),
            DQNMetric(
                "episode_lengths_raw", "episode_indices",
                "dqn_episode_lengths.pdf", "Episode",
                "Episode length (steps)",
                "Smoothed DQN episode length "
                f"({episode_window}-episode moving average)",
                mean_by_episode=True,
                legacy_values="episode_lengths",
                smoothing_window=episode_window,
            ),
            DQNMetric(
                "losses_raw", "loss_steps", "dqn_td_loss.pdf",
                "Environment steps", "TD loss (MSE)",
                "Smoothed DQN TD loss "
                f"({loss_window}-training-step moving average)",
                legacy_values="losses",
                smoothing_window=loss_window,
            ),
            *self.UNSMOOTHED_METRICS,
        )
        self._legacy_warnings = set()
        self._legacy_metrics_used = set()
        self._reconstructed_evaluation_steps = set()
        self._reconstruction_warnings_emitted = set()
        self.logs_dir = LOGS_DIR
        self.output_dir = FIGURES_DIR / "dqn"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = self._discover_experiments()
        self.shared_growth_steps = self._find_shared_growth_steps()
        self._audit_experiments()

    @staticmethod
    def _read_yaml(path):
        try:
            with path.open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
            return data if isinstance(data, dict) else {}
        except (OSError, yaml.YAMLError) as error:
            warnings.warn(f"WARNING: COULD NOT READ {path}: {error}")
            return {}

    @classmethod
    def _color_for(cls, key):
        if key in cls.KNOWN_COLORS:
            return cls.KNOWN_COLORS[key]
        # A stable hash prevents existing methods from changing color when a
        # newly introduced growth method is discovered.
        palette = plt.get_cmap("tab20")
        return palette(zlib.crc32(key.encode("utf-8")) % palette.N)

    def _discover_experiments(self):
        candidates = []
        baseline = self.logs_dir / self.BASELINE_FOLDER
        if baseline.is_dir():
            candidates.append((baseline, "baseline", False))
        for directory in sorted(self.logs_dir.glob(f"{self.GROWTH_PREFIX}*")):
            if directory.is_dir():
                candidates.append((
                    directory,
                    directory.name[len(self.GROWTH_PREFIX):],
                    True,
                ))
        if not candidates:
            raise FileNotFoundError(
                f"No DQN directory matching {self.BASELINE_FOLDER} or "
                f"{self.GROWTH_PREFIX}* in {self.logs_dir}"
            )

        experiments = []
        for directory, folder_key, is_growth in candidates:
            try:
                index = DQNFileIndex(directory)
            except FileNotFoundError as error:
                warnings.warn(f"WARNING: EMPTY DQN DIRECTORY SKIPPED: {error}")
                continue
            configs = {
                seed: self._read_yaml(directory / f"config-{seed}.yaml")
                for seed in index.all_seeds
                if (directory / f"config-{seed}.yaml").exists()
            }
            configured_modes = {
                config.get("growth_mode") for config in configs.values()
                if config.get("growth_mode")
            }
            if is_growth and len(configured_modes) == 1:
                key = next(iter(configured_modes))
            else:
                key = folder_key
            label = self.DISPLAY_LABELS.get(key, key.replace("_", "-"))
            experiments.append(DQNExperiment(
                directory=directory,
                key=key,
                label=label,
                color=self._color_for(key),
                is_growth=is_growth,
                index=index,
                configs=configs,
            ))
        if not experiments:
            raise FileNotFoundError("No non-empty DQN result directory found")
        return experiments

    @staticmethod
    def _growth_events_for_seed(experiment, seed):
        path = experiment.directory / f"growth-events-{seed}.yaml"
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as stream:
                events = yaml.safe_load(stream) or []
        except (OSError, yaml.YAMLError) as error:
            warnings.warn(f"WARNING: COULD NOT READ {path}: {error}")
            return None
        if not isinstance(events, list):
            warnings.warn(f"WARNING: INVALID GROWTH EVENT FILE: {path}")
            return None
        return events

    @classmethod
    def _growth_steps_for_seed(cls, experiment, seed):
        events = cls._growth_events_for_seed(experiment, seed)
        if events is None:
            return None
        return tuple(
            int(event["scheduled_step"])
            for event in events
            if isinstance(event, dict)
            and "scheduled_step" in event
        )

    @classmethod
    def _shared_steps_for_experiment(cls, experiment):
        schedules = [
            steps for seed in experiment.seeds
            if (steps := cls._growth_steps_for_seed(
                experiment, seed
            )) is not None
        ]
        if not schedules or not all(
            schedule == schedules[0] for schedule in schedules[1:]
        ):
            return ()
        return schedules[0]

    def _find_shared_growth_steps(self):
        schedules = []
        for experiment in self.experiments:
            if not experiment.is_growth:
                continue
            for seed in experiment.seeds:
                steps = self._growth_steps_for_seed(experiment, seed)
                if steps is not None:
                    schedules.append(steps)
        if not schedules:
            return ()
        reference = schedules[0]
        if all(schedule == reference for schedule in schedules[1:]):
            return reference
        warnings.warn(
            "WARNING: GROWTH STEPS DIFFER BETWEEN METHODS OR SEEDS; "
            "shared growth markers will not be drawn on comparison plots"
        )
        return ()

    def _audit_experiments(self):
        for experiment in self.experiments:
            issues = []
            for seed in experiment.seeds:
                expected = set(self.COMMON_FILES)
                # Runs predating task metadata remain usable through a
                # deterministic evaluation-axis reconstruction. Once an exact
                # evaluation axis exists, all companion metadata is expected.
                if seed in experiment.index.seeds("evaluation_steps"):
                    expected.update(self.TASK_METADATA_FILES)
                config = experiment.configs.get(seed)
                if config is None:
                    issues.append(f"seed {seed}: missing config-{seed}.yaml")
                    # Plasticity is expected by default in both training scripts.
                    expected.update(self.PLASTICITY_FILES)
                elif config.get("n_plasticity_measurements", 20) > 0:
                    expected.update(self.PLASTICITY_FILES)
                missing = sorted(
                    metric for metric in expected
                    if seed not in experiment.index.seeds(metric)
                    and not (
                        metric in self.LEGACY_METRICS
                        and seed in experiment.index.seeds(
                            self.LEGACY_METRICS[metric]
                        )
                    )
                )
                if missing:
                    issues.append(
                        f"seed {seed}: missing expected metrics: "
                        f"{', '.join(missing)}"
                    )
                if experiment.is_growth and not (
                    experiment.directory / f"growth-events-{seed}.yaml"
                ).exists():
                    issues.append(
                        f"seed {seed}: missing growth-events-{seed}.yaml"
                    )
                elif experiment.is_growth:
                    events = self._growth_events_for_seed(experiment, seed)
                    if events is not None:
                        for event_index, event in enumerate(events):
                            if (
                                not isinstance(event, dict)
                                or event.get("skipped", False)
                                or event.get("neurons_added", 1) <= 0
                            ):
                                continue
                            missing_generation = sorted(
                                name for suffix in self.GENERATION_FILES
                                if seed not in experiment.index.seeds(
                                    name := (
                                        f"generation_{event_index}_{suffix}"
                                    )
                                )
                            )
                            if missing_generation:
                                issues.append(
                                    f"seed {seed}, generation "
                                    f"{event_index}: missing expected metrics: "
                                    f"{', '.join(missing_generation)}"
                                )
            if issues:
                warnings.warn(
                    "\n========== WARNING: INCOMPLETE DQN DATA ==========\n"
                    f"Method: {experiment.label}\n"
                    + "\n".join(issues)
                    + "\n=================================================="
                )

    def _validated_curve(self, experiment, metric, seed):
        values_name = metric.values
        legacy = False
        if seed not in experiment.index.seeds(values_name):
            if (
                metric.legacy_values is None
                or seed not in experiment.index.seeds(metric.legacy_values)
            ):
                return None
            values_name = metric.legacy_values
            legacy = True
        try:
            values = np.asarray(experiment.index.load(values_name, seed))
            if seed in experiment.index.seeds(metric.x):
                x = np.asarray(experiment.index.load(metric.x, seed))
            elif metric.values == "J":
                x = self._reconstruct_evaluation_steps(
                    experiment, seed, len(values)
                )
                if x is None:
                    return None
            else:
                return None
        except (OSError, ValueError) as error:
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}, {metric.ylabel}: "
                f"could not load data ({error}); seed skipped"
            )
            return None
        if x.ndim != 1 or values.ndim != 1 or len(x) != len(values):
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}, {metric.ylabel}: "
                f"invalid shapes x={x.shape}, y={values.shape}; seed skipped"
            )
            return None
        if len(x) == 0:
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}, {metric.ylabel}: "
                "empty arrays; seed skipped"
            )
            return None
        if metric.smoothing_window is not None and not legacy:
            values = moving_average(values, metric.smoothing_window)
        elif legacy:
            warning_key = (experiment.key, seed, values_name)
            if warning_key not in self._legacy_warnings:
                warnings.warn(
                    f"WARNING: {experiment.label} seed {seed}: using legacy "
                    f"pre-smoothed metric '{values_name}'; requested window "
                    "cannot be reapplied because the raw file is missing"
                )
                self._legacy_warnings.add(warning_key)
            self._legacy_metrics_used.add(metric.values)
        finite = np.isfinite(x) & np.isfinite(values)
        if not finite.all():
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}, {metric.ylabel}: "
                f"{np.count_nonzero(~finite)} non-finite points removed"
            )
            x, values = x[finite], values[finite]
        if len(x) == 0 or np.any(np.diff(x) <= 0):
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}, {metric.ylabel}: "
                "x axis is empty, duplicated, or not increasing; seed skipped"
            )
            return None
        return x, values

    @staticmethod
    def _split_budget(total, n_parts):
        quotient, remainder = divmod(total, n_parts)
        return np.asarray([
            quotient + (index < remainder) for index in range(n_parts)
        ])

    def _reconstruct_evaluation_steps(self, experiment, seed, n_values):
        config = experiment.configs.get(seed, {})
        n_timesteps = config.get("n_timesteps")
        n_evaluations = config.get("n_eval_points")
        wind_powers = config.get("wind_powers") or []
        if (
            not isinstance(n_timesteps, int)
            or not isinstance(n_evaluations, int)
            or n_timesteps < 1
            or n_evaluations != n_values
        ):
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}: cannot "
                "reconstruct evaluation_steps from config; seed skipped"
            )
            return None
        n_tasks = len(wind_powers) if config.get("use_curriculum") else 1
        n_tasks = max(1, n_tasks)
        task_steps = self._split_budget(n_timesteps, n_tasks)
        task_evaluations = self._split_budget(n_evaluations, n_tasks)
        blocks = np.concatenate([
            self._split_budget(int(steps), int(evaluations))
            for steps, evaluations in zip(task_steps, task_evaluations)
            if evaluations > 0
        ])
        if len(blocks) != n_values:
            warnings.warn(
                f"WARNING: {experiment.label} seed {seed}: reconstructed "
                "evaluation axis has an inconsistent length; seed skipped"
            )
            return None
        self._reconstructed_evaluation_steps.add((experiment.key, seed))
        return np.cumsum(blocks)

    @staticmethod
    def _align_curves(curves, prefix_alignment):
        if prefix_alignment:
            length = min(len(x) for x, _ in curves)
            reference_x = curves[0][0][:length]
            if not all(
                np.array_equal(x[:length], reference_x)
                for x, _ in curves[1:]
            ):
                return None
            return reference_x, np.stack([y[:length] for _, y in curves])

        common_x = curves[0][0]
        for x, _ in curves[1:]:
            common_x = np.intersect1d(common_x, x, assume_unique=True)
        if len(common_x) == 0:
            return None
        aligned = []
        for x, values in curves:
            indices = np.searchsorted(x, common_x)
            aligned.append(values[indices])
        return common_x, np.stack(aligned)

    @staticmethod
    def _plot_mean_and_std(ax, x, values, experiment):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        lower, upper = mean - std, mean + std
        ax.fill_between(
            x, lower, upper, color=experiment.color, alpha=0.15,
            linewidth=0,
        )
        ax.plot(x, lower, color=experiment.color, linewidth=0.55, alpha=0.5)
        ax.plot(x, upper, color=experiment.color, linewidth=0.55, alpha=0.5)
        ax.plot(
            x, mean, color=experiment.color, linewidth=2,
            label=experiment.label,
        )

    def _add_shared_growth_markers(self, ax):
        for step in self.shared_growth_steps:
            ax.axvline(
                step, color="indigo", linestyle="--", linewidth=0.8,
                alpha=0.65, zorder=0,
            )

    def plot_metric(self, metric):
        self._legacy_metrics_used.discard(metric.values)
        figure, ax = plt.subplots(figsize=(10, 4.5))
        plotted = False
        applicable = False
        for experiment in self.experiments:
            if metric.values == "plasticity" and experiment.configs and all(
                config.get("n_plasticity_measurements", 20) == 0
                for config in experiment.configs.values()
            ):
                continue
            applicable = True
            value_seeds = experiment.index.seeds(metric.values)
            if metric.legacy_values is not None:
                value_seeds |= experiment.index.seeds(metric.legacy_values)
            if metric.values == "J":
                complete_seeds = sorted(value_seeds)
            else:
                complete_seeds = sorted(
                    value_seeds & experiment.index.seeds(metric.x)
                )
            curves = []
            for seed in complete_seeds:
                curve = self._validated_curve(experiment, metric, seed)
                if curve is not None:
                    curves.append(curve)
            if (
                metric.values == "J"
                and experiment.key
                not in self._reconstruction_warnings_emitted
            ):
                reconstructed_seeds = sorted(
                    seed for key, seed
                    in self._reconstructed_evaluation_steps
                    if key == experiment.key
                )
                if reconstructed_seeds:
                    warnings.warn(
                        f"WARNING: {experiment.label}: evaluation_steps "
                        "reconstructed from legacy configs for seeds "
                        f"{reconstructed_seeds}"
                    )
                    self._reconstruction_warnings_emitted.add(
                        experiment.key
                    )
            if not curves:
                continue
            aligned = self._align_curves(curves, metric.mean_by_episode)
            if aligned is None:
                warnings.warn(
                    f"WARNING: {experiment.label}, {metric.ylabel}: seeds "
                    "have no compatible x axis; method skipped"
                )
                continue
            self._plot_mean_and_std(ax, *aligned, experiment)
            plotted = True

        if not plotted:
            plt.close(figure)
            if not applicable:
                return
            warnings.warn(
                f"WARNING: NO VALID DATA FOR {metric.ylabel}; plot skipped"
            )
            return
        if metric.xlabel == "Environment steps":
            self._add_shared_growth_markers(ax)
        title = metric.title
        if metric.values in self._legacy_metrics_used:
            title = re.sub(r" \([^()]+ moving average\)$", "", title)
            title += " (legacy smoothing window unavailable)"
        finish_figure(
            figure, ax, self.output_dir / metric.filename,
            metric.xlabel, metric.ylabel, title, legend_columns=1,
        )

    def _generation_indices(self, experiment):
        indices = set()
        pattern = re.compile(r"generation_(\d+)_steps")
        for metric_name in experiment.index.files:
            match = pattern.fullmatch(metric_name)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def plot_generation_metrics(self, experiment):
        generations = self._generation_indices(experiment)
        if not generations:
            return
        method_dir = self.output_dir / "growth" / experiment.key.replace("-", "_")
        method_dir.mkdir(parents=True, exist_ok=True)
        generation_colors = plt.get_cmap("viridis")
        denominator = max(1, len(generations) - 1)
        for metric_suffix, ylabel in self.GENERATION_METRICS.items():
            figure, ax = plt.subplots(figsize=(10, 4.5))
            plotted = False
            for position, generation in enumerate(generations):
                metric = DQNMetric(
                    f"generation_{generation}_{metric_suffix}",
                    f"generation_{generation}_steps", "",
                    "Environment steps", ylabel, "",
                )
                seeds = sorted(
                    experiment.index.seeds(metric.values)
                    & experiment.index.seeds(metric.x)
                )
                curves = [
                    curve for seed in seeds
                    if (curve := self._validated_curve(
                        experiment, metric, seed
                    )) is not None
                ]
                if not curves:
                    continue
                aligned = self._align_curves(curves, False)
                if aligned is None:
                    warnings.warn(
                        f"WARNING: {experiment.label}, generation "
                        f"{generation}: incompatible axes; skipped"
                    )
                    continue
                x, values = aligned
                color = generation_colors(position / denominator)
                mean, std = values.mean(axis=0), values.std(axis=0)
                ax.fill_between(
                    x, mean - std, mean + std, color=color, alpha=0.13,
                    linewidth=0,
                )
                ax.plot(
                    x, mean, color=color, linewidth=1.7,
                    label=f"Cohort {generation + 1}",
                )
                plotted = True
            if not plotted:
                plt.close(figure)
                continue
            for step in self._shared_steps_for_experiment(experiment):
                ax.axvline(
                    step, color="indigo", linestyle="--",
                    linewidth=0.8, alpha=0.65, zorder=0,
                )
            finish_figure(
                figure, ax, method_dir / f"dqn_{metric_suffix}.pdf",
                "Environment steps", ylabel,
                f"{experiment.label}: {ylabel}",
                legend_columns=1,
            )

    def plot_architecture_evolution(self):
        width_figure, width_ax = plt.subplots(figsize=(10, 4.5))
        neurons_figure, neurons_ax = plt.subplots(figsize=(10, 4.5))
        width_plotted = False
        neurons_plotted = False
        skipped_plotted = False
        zero_plotted = False

        for experiment in self.experiments:
            if not experiment.is_growth:
                continue
            seed_events = []
            for seed in experiment.seeds:
                events = self._growth_events_for_seed(experiment, seed)
                if events is not None:
                    valid = [
                        event for event in events
                        if isinstance(event, dict)
                        and "scheduled_step" in event
                        and "hidden_before" in event
                        and "hidden_after" in event
                    ]
                    if valid:
                        seed_events.append(valid)
            if not seed_events:
                continue

            common_steps = set(
                int(event["scheduled_step"]) for event in seed_events[0]
            )
            for events in seed_events[1:]:
                common_steps &= {
                    int(event["scheduled_step"]) for event in events
                }
            common_steps = sorted(common_steps)
            if not common_steps:
                warnings.warn(
                    f"WARNING: {experiment.label}: no common architecture "
                    "events across seeds; diagnostics skipped"
                )
                continue

            event_maps = [{
                int(event["scheduled_step"]): event for event in events
            } for events in seed_events]
            initial_widths = np.asarray([
                event_maps[index][common_steps[0]]["hidden_before"]
                for index in range(len(event_maps))
            ], dtype=float)
            widths = np.asarray([
                [event_map[step]["hidden_after"] for step in common_steps]
                for event_map in event_maps
            ], dtype=float)
            width_values = np.column_stack((initial_widths, widths))
            width_x = np.asarray([0, *common_steps])
            width_mean = width_values.mean(axis=0)
            width_std = width_values.std(axis=0)
            width_ax.fill_between(
                width_x, width_mean - width_std, width_mean + width_std,
                step="post", color=experiment.color, alpha=0.15,
                linewidth=0,
            )
            width_ax.step(
                width_x, width_mean - width_std, where="post",
                color=experiment.color, linewidth=0.55, alpha=0.5,
            )
            width_ax.step(
                width_x, width_mean + width_std, where="post",
                color=experiment.color, linewidth=0.55, alpha=0.5,
            )
            width_ax.step(
                width_x, width_mean, where="post", color=experiment.color,
                linewidth=2, label=experiment.label,
            )
            width_plotted = True

            added = np.asarray([
                [event_map[step].get("neurons_added", 0)
                 for step in common_steps]
                for event_map in event_maps
            ], dtype=float)
            added_mean, added_std = added.mean(axis=0), added.std(axis=0)
            neurons_ax.fill_between(
                common_steps, added_mean - added_std,
                added_mean + added_std, color=experiment.color,
                alpha=0.15, linewidth=0,
            )
            neurons_ax.plot(
                common_steps, added_mean - added_std,
                color=experiment.color, linewidth=0.55, alpha=0.5,
            )
            neurons_ax.plot(
                common_steps, added_mean + added_std,
                color=experiment.color, linewidth=0.55, alpha=0.5,
            )
            neurons_ax.plot(
                common_steps, added_mean, color=experiment.color,
                linewidth=2, marker="o", markersize=4,
                label=experiment.label,
            )
            neurons_plotted = True

            for step_index, step in enumerate(common_steps):
                events = [event_map[step] for event_map in event_maps]
                if any(event.get("skipped", False) for event in events):
                    neurons_ax.scatter(
                        step, 0, marker="x", s=55, linewidths=1.5,
                        color=experiment.color, zorder=4,
                        label="Skipped event" if not skipped_plotted else None,
                    )
                    skipped_plotted = True
                elif added_mean[step_index] == 0:
                    neurons_ax.scatter(
                        step, 0, marker="o", s=42, facecolors="none",
                        edgecolors=experiment.color, linewidths=1.2, zorder=4,
                        label=(
                            "Executed, zero neurons"
                            if not zero_plotted else None
                        ),
                    )
                    zero_plotted = True

        if width_plotted:
            finish_figure(
                width_figure, width_ax,
                self.output_dir / "dqn_network_width.pdf",
                "Scheduled environment step", "Hidden-layer width",
                "DQN architecture evolution", legend_columns=1,
            )
        else:
            plt.close(width_figure)
        if neurons_plotted:
            finish_figure(
                neurons_figure, neurons_ax,
                self.output_dir / "dqn_neurons_added.pdf",
                "Scheduled environment step", "Neurons added",
                "DQN growth decisions", legend_columns=1,
            )
        else:
            plt.close(neurons_figure)

    def run(self):
        print("Detected DQN experiments:")
        for experiment in self.experiments:
            print(f"  {experiment.label}: seeds {experiment.seeds}")
        for metric in self.metrics:
            self.plot_metric(metric)
        for experiment in self.experiments:
            if experiment.is_growth:
                self.plot_generation_metrics(experiment)
        self.plot_architecture_evolution()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "algorithm", choices=("fqi", "dqn"),
        help="Type of experiment to visualize",
    )
    parser.add_argument(
        "--reward-window", type=int, default=500,
        help="Moving-average window for per-step training rewards",
    )
    parser.add_argument(
        "--episode-window", type=int, default=50,
        help="Moving-average window for episodic returns and lengths",
    )
    parser.add_argument(
        "--loss-window", type=int, default=50,
        help="Moving-average window for per-training-step TD losses",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualizer = FQIVisualizer() if args.algorithm == "fqi" else DQNVisualizer(
        reward_window=args.reward_window,
        episode_window=args.episode_window,
        loss_window=args.loss_window,
    )
    visualizer.run()
    print(f"Saved figures to {visualizer.output_dir}")
