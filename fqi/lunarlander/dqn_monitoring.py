"""Representation monitoring shared by LunarLander DQN experiments."""

from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from fqi.utils.growing_network import (
    feature_rank,
    measure_plasticity,
    normalized_bellman_residual_correlation,
    principal_angle_cosines,
    srank,
)


def _preserve_random_states(function):
    """Keep diagnostic computations from changing the training trajectory."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else None
        )
        try:
            return function(*args, **kwargs)
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return wrapped


def replay_batch_to_tensors(agent, replay_batch):
    """Convert replay transitions and compute frozen DQN TD targets."""
    states, actions, rewards, next_states, absorbing, _ = replay_batch
    if agent._clip_reward:
        rewards = np.clip(rewards, -1, 1)
    td_targets = rewards + agent.mdp_info.gamma * agent._next_q(
        next_states, absorbing
    )
    device = next(_active_networks(agent)[-1].parameters()).device
    return (
        torch.as_tensor(states, dtype=torch.float32, device=device),
        torch.as_tensor(actions.reshape(-1), dtype=torch.long, device=device),
        torch.as_tensor(td_targets, dtype=torch.float32, device=device),
    )


def _active_networks(agent):
    """Return the neural networks contributing to the current Q-function."""
    model = agent.approximator.model
    if hasattr(model, "_model"):  # MushroomRL Ensemble
        last = getattr(agent, "_curriculum_idx", len(model) - 1)
        return [model[idx].network for idx in range(last + 1)]
    return [model.network]


class DQNMetricsMonitor:
    """Monitor features on fixed data and plasticity on fresh replay data."""

    def __init__(
        self,
        n_eval_points,
        monitoring_batch_size,
        n_plasticity_measurements,
        plasticity_n_samples,
        plasticity_n_steps,
        plasticity_n_tasks,
        learning_rate,
    ):
        if monitoring_batch_size < 1 or plasticity_n_samples < 1:
            raise ValueError("monitoring batch sizes must be positive")
        if not 0 <= n_plasticity_measurements <= n_eval_points:
            raise ValueError(
                "n_plasticity_measurements must be between 0 and "
                "n_eval_points"
            )
        if plasticity_n_steps < 1 or plasticity_n_tasks < 1:
            raise ValueError("plasticity_n_steps and n_tasks must be positive")

        self.monitoring_batch_size = monitoring_batch_size
        self.plasticity_n_samples = plasticity_n_samples
        self.plasticity_n_steps = plasticity_n_steps
        self.plasticity_n_tasks = plasticity_n_tasks
        self.learning_rate = learning_rate
        self.plasticity_checkpoints = {
            int(round(k * n_eval_points / n_plasticity_measurements))
            for k in range(1, n_plasticity_measurements + 1)
        } if n_plasticity_measurements else set()

        self.fixed_batch = None
        self.steps = []
        self.task_indices = []
        self.ranks = []
        self.rank_ratios = []
        self.sranks = []
        self.srank_ratios = []
        self.plasticity_steps = []
        self.plasticity_task_indices = []
        self.plasticities = []
        self.generations = []

    def start_task(self):
        """Discard monitoring transitions belonging to the previous MDP."""
        self.fixed_batch = None

    def _ensure_fixed_batch(self, agent):
        if self.fixed_batch is None:
            self.fixed_batch = agent._replay_memory.get(
                self.monitoring_batch_size
            )

    @staticmethod
    def _features_and_residual(agent, replay_batch):
        states, actions, td_targets = replay_batch_to_tensors(
            agent, replay_batch
        )
        networks = _active_networks(agent)
        training_modes = [network.training for network in networks]
        for network in networks:
            network.eval()
        with torch.no_grad():
            features = torch.cat(
                [network.encode(states) for network in networks], dim=1
            ).cpu().numpy()
            q_values = sum(network(states) for network in networks)
            predictions = (
                q_values
                .gather(1, actions.reshape(-1, 1))
                .squeeze(1)
                .cpu()
                .numpy()
            )
        for network, was_training in zip(networks, training_modes):
            network.train(was_training)
        residual = (td_targets.cpu().numpy() - predictions).reshape(-1, 1)
        return features, residual

    @_preserve_random_states
    def register_growth(self, agent, event_index, start, end, step):
        """Register and immediately measure one newly added neuron cohort."""
        if end <= start:
            return
        self._ensure_fixed_batch(agent)
        generation = {
            "event_index": int(event_index),
            "start": int(start),
            "end": int(end),
            "growth_step": int(step),
            "steps": [],
            "angle_min": [],
            "angle_max": [],
            "angle_mean": [],
            "brc_normalized": [],
        }
        self.generations.append(generation)
        features, residual = self._features_and_residual(
            agent, self.fixed_batch
        )
        self._record_generation(generation, features, residual, step)

    @staticmethod
    def _record_generation(generation, features, residual, step):
        if generation["steps"] and generation["steps"][-1] == int(step):
            return
        start = generation["start"]
        end = generation["end"]
        phi_old = features[:, :start]
        phi_new = features[:, start:end]
        if phi_old.shape[1] == 0 or phi_new.shape[1] == 0:
            return
        cosines = principal_angle_cosines(phi_old, phi_new)
        angle_min = float(cosines.min()) if cosines.size else float("nan")
        angle_max = float(cosines.max()) if cosines.size else float("nan")
        angle_mean = float(cosines.mean()) if cosines.size else float("nan")
        generation["steps"].append(int(step))
        generation["angle_min"].append(angle_min)
        generation["angle_max"].append(angle_max)
        generation["angle_mean"].append(angle_mean)
        generation["brc_normalized"].append(
            normalized_bellman_residual_correlation(
                phi_new, residual, epsilon=1e-12
            )
        )

    @_preserve_random_states
    def monitor_evaluation(
        self, agent, step, evaluation_index, task_index=0
    ):
        if not agent._replay_memory.initialized:
            return
        self._ensure_fixed_batch(agent)
        features, residual = self._features_and_residual(
            agent, self.fixed_batch
        )
        rank, singular_values = feature_rank(features)
        effective_rank = srank(singular_values)
        feature_dimension = features.shape[1]
        self.steps.append(int(step))
        self.task_indices.append(int(task_index))
        self.ranks.append(rank)
        self.rank_ratios.append(rank / feature_dimension)
        self.sranks.append(effective_rank)
        self.srank_ratios.append(effective_rank / feature_dimension)

        for generation in self.generations:
            self._record_generation(generation, features, residual, step)

        if evaluation_index in self.plasticity_checkpoints:
            fresh_batch = agent._replay_memory.get(self.plasticity_n_samples)
            states = replay_batch_to_tensors(agent, fresh_batch)[0]
            network = _active_networks(agent)[-1]
            plasticity = measure_plasticity(
                network,
                states,
                optimizer_class=optim.Adam,
                optimizer_params={"lr": self.learning_rate},
                n_steps=self.plasticity_n_steps,
                n_tasks=self.plasticity_n_tasks,
            )
            self.plasticity_steps.append(int(step))
            self.plasticity_task_indices.append(int(task_index))
            self.plasticities.append(plasticity)

    def save(self, output_dir, seed):
        output_dir = Path(output_dir)
        arrays = {
            "monitoring-steps": self.steps,
            "monitoring-task-indices": self.task_indices,
            "feature-rank": self.ranks,
            "feature-rank-ratio": self.rank_ratios,
            "feature-srank": self.sranks,
            "feature-srank-ratio": self.srank_ratios,
            "plasticity-steps": self.plasticity_steps,
            "plasticity-task-indices": self.plasticity_task_indices,
            "plasticity": self.plasticities,
        }
        for name, values in arrays.items():
            np.save(
                output_dir / f"{name}-{seed}.npy",
                np.asarray(values),
            )

        for generation in self.generations:
            prefix = f"generation-{generation['event_index']}"
            generation_arrays = {
                "bounds": [generation["start"], generation["end"]],
                "growth-step": [generation["growth_step"]],
                "steps": generation["steps"],
                "principal-angle-min": generation["angle_min"],
                "principal-angle-max": generation["angle_max"],
                "principal-angle-mean": generation["angle_mean"],
                "brc-normalized": generation["brc_normalized"],
            }
            for name, values in generation_arrays.items():
                np.save(
                    output_dir / f"{prefix}-{name}-{seed}.npy",
                    np.asarray(values),
                )
