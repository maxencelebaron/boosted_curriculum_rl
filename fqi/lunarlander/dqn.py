"""DQN extensions used by the LunarLander experiments."""

import numpy as np

from mushroom_rl.algorithms.value.dqn import DQN


class LunarLanderDQN(DQN):
    """DQN performing several gradient steps per replay-buffer update."""

    def __init__(self, *args, gradient_steps=1, **kwargs):
        self._gradient_steps = gradient_steps
        self.training_loss_steps = []
        self.training_losses = []
        self._samples_seen = 0
        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        self._samples_seen += len(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._gradient_steps):
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                q_next = self._next_q(next_state, absorbing)
                target = reward + self.mdp_info.gamma * q_next
                self.approximator.fit(
                    state, action, target, **self._fit_params
                )
                self.training_loss_steps.append(self._samples_seen)
                self.training_losses.append(
                    float(self.approximator.model.loss_fit)
                )


class BoostedLunarLanderDQN(DQN):
    """Boosted DQN using one residual Q-network per curriculum task."""

    def __init__(self, *args, gradient_steps=1, **kwargs):
        self._gradient_steps = gradient_steps
        self._curriculum_idx = 0
        self.training_loss_steps = []
        self.training_losses = []
        self._samples_seen = 0
        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        self._samples_seen += len(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._gradient_steps):
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                if self._curriculum_idx > 0:
                    self._predict_params["idx"] = np.arange(
                        self._curriculum_idx
                    )
                    previous_q = self.approximator.predict(
                        state,
                        action.astype(np.int64),
                        **self._predict_params,
                    )
                    self._predict_params["idx"] = np.arange(
                        self._curriculum_idx + 1
                    )
                else:
                    previous_q = 0.0

                q_next = self._next_q(next_state, absorbing)
                target = (
                    reward + self.mdp_info.gamma * q_next - previous_q
                )
                self.approximator.fit(
                    state, action, target, **self._fit_params
                )
                self.training_loss_steps.append(self._samples_seen)
                self.training_losses.append(
                    float(self.approximator.model[self._curriculum_idx].loss_fit)
                )

    def _update_target(self):
        self.target_approximator[self._curriculum_idx].set_weights(
            self.approximator[self._curriculum_idx].get_weights()
        )

    def set_curriculum_idx_and_reset(self, curriculum_idx):
        self._curriculum_idx = curriculum_idx
        self._fit_params["idx"] = curriculum_idx
        self._predict_params["idx"] = np.arange(curriculum_idx + 1)
        self.policy._predict_params["idx"] = np.arange(curriculum_idx + 1)


class SingleBoostedLunarLanderDQN(DQN):
    """Train one residual DQN on top of a fixed previous Q-function."""

    def __init__(self, *args, prev_q, gradient_steps=1, **kwargs):
        self.prev_q = prev_q
        self._gradient_steps = gradient_steps
        self.training_loss_steps = []
        self.training_losses = []
        self._samples_seen = 0
        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        self._samples_seen += len(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._gradient_steps):
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                previous_q = self.prev_q.predict(
                    state, action.astype(np.int64)
                )
                previous_next_q = self.prev_q.predict(next_state)
                q_next = self._next_q(next_state, absorbing)
                target = (
                    reward
                    + self.mdp_info.gamma * (q_next + previous_next_q)
                    - previous_q
                )
                self.approximator.fit(
                    state, action, target, **self._fit_params
                )
                self.training_loss_steps.append(self._samples_seen)
                self.training_losses.append(
                    float(self.approximator.model.loss_fit)
                )

    def _update_target(self):
        self.target_approximator.set_weights(
            self.approximator.get_weights()
        )
