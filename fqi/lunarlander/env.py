import gymnasium as gym
import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces


class LunarLander(Environment):
    """MushroomRL adapter around Gymnasium's LunarLander-v3."""

    def __init__(self, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5, gamma=0.99, horizon=1000):
        self._seed = None
        self._env = gym.make(
            "LunarLander-v3",
            continuous=False,
            gravity=gravity,  # to be within 0 and -12, Default is -10.0
            enable_wind=enable_wind,
            wind_power=wind_power,  # The recommended value for `wind_power` is between 0.0 and 20.0.
            turbulence_power=turbulence_power,   # The recommended value for `turbulence_power` is between 0.0 and 2.0.
        )
        observation_space = spaces.Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
        )
        action_space = spaces.Discrete(self._env.action_space.n)
        super().__init__(MDPInfo(observation_space, action_space, gamma, horizon))

    def seed(self, seed):
        self._seed = seed

    def reset(self, state=None):
        if state is not None:
            raise ValueError("LunarLander does not support arbitrary initial states")
        observation, _ = self._env.reset(seed=self._seed)
        self._seed = None
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(
            int(np.asarray(action).item())
        )
        return observation, reward, terminated or truncated, info

    def render(self, record=False):
        del record
        return self._env.render()

    def stop(self):
        self._env.close()
