import numpy as np
from tqdm import trange

from mushroom_rl.utils.dataset import parse_dataset

from fqi.fqi import BoostedFQI, FQI

# NeuralFQI is identical to FQI: FQI.fit already forwards **self._fit_params
# to the approximator, so Q_Network.fit receives lr/n_epochs/batch_size.
NeuralFQI = FQI


class BoostedNeuralFQI(BoostedFQI):
    """
    BoostedFQI variant for neural weak learners.

    The only difference from BoostedFQI is that approximator.fit receives
    **self._fit_params (lr, n_epochs, batch_size), which BoostedFQI does not
    forward because sklearn estimators do not need per-fit hyper-parameters.
    """

    def fit(self, x, callback=None, n_iter=None):
        if n_iter is None:
            n_iter = self._n_iterations

        state, action, reward, next_state, absorbing, _ = parse_dataset(x)

        if self._curriculum_idx > 0:
            prev_q = self.approximator.predict(
                state,
                action.astype(np.int64),
                idx=np.arange(self._curriculum_idx)
            )
            prev_q_next_state = self.approximator.predict(
                next_state,
                idx=np.arange(self._curriculum_idx)
            )
        else:
            prev_q = np.zeros(state.shape[0])
            prev_q_next_state = np.zeros(
                (next_state.shape[0], self.approximator.n_actions)
            )

        for _ in trange(
            n_iter,
            dynamic_ncols=True,
            disable=self._quiet,
            leave=False
        ):
            q_next_state = prev_q_next_state + self.approximator.predict(
                next_state, idx=self._curriculum_idx)

            if np.any(absorbing):
                q_next_state *= 1 - absorbing.reshape(-1, 1)

            max_q_next_state = np.max(q_next_state, axis=1)
            target = reward + self.mdp_info.gamma * max_q_next_state - prev_q
            self.approximator.fit(
                state,
                action,
                target,
                idx=self._curriculum_idx,
                **self._fit_params
            )

            if callback is not None:
                callback()
