import copy
from mushroom_rl.algorithms.value.dqn import DQN
from mushroom_rl.approximators.parametric.torch_approximator import *


class CarOnHillDQN(DQN):

    def __init__(self, *args, gradient_steps=1, **kwargs):
        self._gradient_steps = gradient_steps
        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._gradient_steps):
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                q_next = self._next_q(next_state, absorbing)
                q = reward + self.mdp_info.gamma * q_next

                self.approximator.fit(state, action, q, **self._fit_params)


class BoostedCarOnHillDQN(DQN):

    def __init__(self, *args, gradient_steps=1, **kwargs):
        self._gradient_steps = gradient_steps
        self._curriculum_idx = 0
        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._gradient_steps):
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                if self._curriculum_idx > 0:
                    self._predict_params['idx'] = np.arange(self._curriculum_idx)
                    prev_qs = self.approximator.predict(state, action.astype(np.int64), **self._predict_params)
                    self._predict_params['idx'] = np.arange(self._curriculum_idx + 1)
                else:
                    prev_qs = 0.

                q_next = self._next_q(next_state, absorbing)
                q = reward + self.mdp_info.gamma * q_next - prev_qs

                self.approximator.fit(state, action, q, **self._fit_params)

    def _update_target(self):
        self.target_approximator[self._curriculum_idx].set_weights(
            self.approximator[self._curriculum_idx].get_weights()
        )

    def set_curriculum_idx_and_reset(self, curriculum_idx):
        self._curriculum_idx = curriculum_idx
        self._fit_params['idx'] = curriculum_idx
        self._predict_params['idx'] = np.arange(curriculum_idx + 1)
        self.policy._predict_params['idx'] = np.arange(curriculum_idx + 1)


class SingleBoostedDQN(DQN):

    def __init__(self, *args, prev_q, gradient_steps=1, **kwargs):
        self.prev_q = prev_q
        self._gradient_steps = gradient_steps

        super().__init__(*args, **kwargs)

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            for _ in range(self._gradient_steps):
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                if self._clip_reward:
                    reward = np.clip(reward, -1, 1)

                prev_qs = self.prev_q.predict(state, action.astype(np.int64))
                prev_qs_next = self.prev_q.predict(next_state)
                q_next = self._next_q(next_state, absorbing)
                q = reward + self.mdp_info.gamma * (q_next + prev_qs_next) - prev_qs

                self.approximator.fit(state, action, q, **self._fit_params)

    def _update_target(self):
        self.target_approximator.set_weights(self.approximator.get_weights())
