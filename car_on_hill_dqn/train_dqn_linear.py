from joblib import Parallel, delayed

import os
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.environments.car_on_hill import CarOnHill

from car_on_hill_dqn.dqn import CarOnHillDQN
from car_on_hill_dqn.network import Network

torch.set_num_threads(1)

N_TIMESTEPS = 120000
N_EVAL_POINTS = 60
TRAIN_FREQ = 16
GRADIENT_STEPS = 8
EXPLORATION_FRACTION = 0.2
EXPLORATION_FINAL_EPS = 0.07
LEARNING_RATE = 4e-3
BATCH_SIZE = 128
BUFFER_SIZE = 10000
LEARNING_STARTS = 1000
TARGET_UPDATE_INTERVAL = 600


def train_dqn(seed, performance_path):
    np.random.seed(seed)
    mdp = CarOnHill()

    optimizer = {'class': optim.Adam, 'params': dict(lr=LEARNING_RATE)}

    approximator_params = dict(
        network=Network,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        loss=F.mse_loss,
        optimizer=optimizer,
        use_cuda=True,
    )

    algorithm_params = dict(
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_INTERVAL,
        initial_replay_size=LEARNING_STARTS,
        max_replay_size=BUFFER_SIZE,
    )

    n_steps_per_fit = TRAIN_FREQ
    steps_per_eval = N_TIMESTEPS // N_EVAL_POINTS
    n_explore = int(N_TIMESTEPS * EXPLORATION_FRACTION)

    epsilon = LinearParameter(value=1.0, threshold_value=EXPLORATION_FINAL_EPS, n=n_explore)
    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon)

    agent = CarOnHillDQN(mdp.info, pi, TorchApproximator,
                         gradient_steps=GRADIENT_STEPS,
                         approximator_params=approximator_params,
                         **algorithm_params)
    core = Core(agent, mdp)

    performances = []
    for _ in range(N_EVAL_POINTS):
        pi.set_epsilon(epsilon)
        core.learn(n_steps=steps_per_eval, n_steps_per_fit=n_steps_per_fit)
        pi.set_epsilon(test_epsilon)
        performances.append(np.mean(compute_J(core.evaluate(n_episodes=50), gamma=mdp.info.gamma)))

    np.save(performance_path, np.array(performances))


def experiment(seed):
    log_dir = "logs/dqn_car_on_hill"
    os.makedirs(log_dir, exist_ok=True)
    train_dqn(seed, os.path.join(log_dir, "performances-%d.npy" % seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-exp", type=int, default=20)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    Parallel(n_jobs=args.n_jobs)(delayed(experiment)(k) for k in range(args.n_exp))
