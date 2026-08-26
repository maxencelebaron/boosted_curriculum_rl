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
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, pad=title_pad)
    ax.grid(alpha=0.3)
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

    METRICS = (
        DQNMetric(
            "J",
            "evaluation_steps",
            "dqn_evaluation_returns.pdf",
            "Environment steps",
            "Cumulative discounted return",
            "DQN evaluation performance",
        ),
        DQNMetric(
            "training_rewards",
            "training_reward_steps",
            "dqn_training_rewards.pdf",
            "Environment steps",
            "Immediate reward",
            "DQN training reward (rolling mean)",
        ),
        DQNMetric(
            "episode_returns",
            "episode_indices",
            "dqn_episode_returns.pdf",
            "Episode",
            "Episode return",
            "DQN episode return (rolling mean)",
            mean_by_episode=True,
        ),
        DQNMetric(
            "episode_lengths",
            "episode_indices",
            "dqn_episode_lengths.pdf",
            "Episode",
            "Episode length (steps)",
            "DQN episode length (rolling mean)",
            mean_by_episode=True,
        ),
        DQNMetric(
            "losses",
            "loss_steps",
            "dqn_td_loss.pdf",
            "Environment steps",
            "TD loss (MSE)",
            "DQN TD loss (rolling mean)",
        ),
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

    # Files produced for every seed by the current training scripts. Raw
    # arrays are audited but deliberately not plotted.
    COMMON_FILES = {
        "J", "evaluation_steps", "evaluation_task_indices",
        "task_boundaries", "task_wind_powers", "task_timesteps",
        "task_evaluations", "training_reward_steps",
        "training_rewards_raw", "training_rewards", "episode_indices",
        "episode_returns_raw", "episode_returns", "episode_lengths_raw",
        "episode_lengths", "loss_steps", "losses_raw", "losses",
        "monitoring_steps", "monitoring_task_indices", "feature_rank",
        "feature_rank_ratio", "feature_srank", "feature_srank_ratio",
    }
    PLASTICITY_FILES = {
        "plasticity_steps", "plasticity_task_indices", "plasticity",
    }
    GENERATION_METRICS = {
        "principal_angle_min": "Minimum principal-angle cosine",
        "principal_angle_max": "Maximum principal-angle cosine",
        "principal_angle_mean": "Mean principal-angle cosine",
        "brc_normalized": "Normalized Bellman residual correlation",
    }
    GENERATION_FILES = {
        "bounds", "growth_step", "steps", *GENERATION_METRICS,
    }

    def __init__(self):
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
            label = "Baseline" if not is_growth else key.replace("_", "-")
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
            int(event.get("actual_step", event.get("scheduled_step")))
            for event in events
            if isinstance(event, dict)
            and not event.get("skipped", False)
            and event.get("neurons_added", 1) > 0
            and ("actual_step" in event or "scheduled_step" in event)
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

    @staticmethod
    def _validated_curve(experiment, metric, seed):
        try:
            x = np.asarray(experiment.index.load(metric.x, seed))
            values = np.asarray(experiment.index.load(metric.values, seed))
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
            label=f"{experiment.label} ({len(values)} seeds)",
        )

    def _add_shared_growth_markers(self, ax):
        for step in self.shared_growth_steps:
            ax.axvline(
                step, color="indigo", linestyle="--", linewidth=0.8,
                alpha=0.65, zorder=0,
            )

    def plot_metric(self, metric):
        figure, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        applicable = False
        for experiment in self.experiments:
            if metric.values == "plasticity" and experiment.configs and all(
                config.get("n_plasticity_measurements", 20) == 0
                for config in experiment.configs.values()
            ):
                continue
            applicable = True
            complete_seeds = sorted(
                experiment.index.seeds(metric.values)
                & experiment.index.seeds(metric.x)
            )
            curves = []
            for seed in complete_seeds:
                curve = self._validated_curve(experiment, metric, seed)
                if curve is not None:
                    curves.append(curve)
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
        self._add_shared_growth_markers(ax)
        finish_figure(
            figure, ax, self.output_dir / metric.filename,
            metric.xlabel, metric.ylabel, metric.title,
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
            figure, ax = plt.subplots(figsize=(8, 4.5))
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
                    label=f"Generation {generation + 1}",
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
            )

    def run(self):
        print("Detected DQN experiments:")
        for experiment in self.experiments:
            print(f"  {experiment.label}: seeds {experiment.seeds}")
        for metric in self.METRICS:
            self.plot_metric(metric)
        for experiment in self.experiments:
            if experiment.is_growth:
                self.plot_generation_metrics(experiment)


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
