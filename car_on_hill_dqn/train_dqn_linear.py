from joblib import Parallel, delayed

import os
import torch
import pathlib
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
from fqi.car_on_hill.solver import solve_car_on_hill

torch.set_num_threads(1)

N_TIMESTEPS = 120000
N_EVAL_POINTS = 20
TRAIN_FREQ = 16
GRADIENT_STEPS = 8
EXPLORATION_FRACTION = 0.2
EXPLORATION_FINAL_EPS = 0.07
LEARNING_RATE = 4e-3
BATCH_SIZE = 128
BUFFER_SIZE = 10000
LEARNING_STARTS = 1000
TARGET_UPDATE_INTERVAL = 600


def train_dqn(seed, log_dir):
    np.random.seed(seed)
    mdp = CarOnHill()

    # Test grid
    test_states_0 = np.linspace(
        mdp.info.observation_space.low[0],
        mdp.info.observation_space.high[0], 10)
    test_states_1 = np.linspace(
        mdp.info.observation_space.low[1],
        mdp.info.observation_space.high[1], 10)
    test_states = []
    for s0 in test_states_0:
        for s1 in test_states_1:
            test_states += [s0, s1]
    test_states = np.array([test_states]).repeat(2, 0).reshape(-1, 2)
    test_actions = np.array(
        [np.zeros(len(test_states) // 2),
         np.ones(len(test_states) // 2)]).reshape(-1, 1).astype(int)

    # Q*
    name = '%1.3f' % mdp._m
    q_path = 'data/test_q_%s.npy' % name
    try:
        test_q = np.load(q_path)
    except FileNotFoundError:
        pathlib.Path('data').mkdir(parents=True, exist_ok=True)
        test_q = np.array(solve_car_on_hill(mdp, test_states, test_actions, mdp.info.gamma))
        np.save(q_path, test_q)

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

    js, diff_qs, bias_sas, bias_maxs = [], [], [], []
    n_states = len(test_states) // 2

    for _ in range(N_EVAL_POINTS):
        pi.set_epsilon(epsilon)
        core.learn(n_steps=steps_per_eval, n_steps_per_fit=n_steps_per_fit)

        pi.set_epsilon(test_epsilon)
        test_dataset = core.evaluate(initial_states=test_states, quiet=True)

        js.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))

        qs = agent.approximator.predict(test_states, test_actions)
        diff_qs.append(np.linalg.norm(qs - test_q, ord=1) / len(qs))
        bias_sas.append(np.mean(qs - test_q))
        q_hat_max = np.maximum(qs[:n_states], qs[n_states:])
        q_star_max = np.maximum(test_q[:n_states], test_q[n_states:])
        bias_maxs.append(np.mean(q_hat_max - q_star_max))

    np.save(os.path.join(log_dir, 'J-%d.npy' % seed), js)
    np.save(os.path.join(log_dir, 'Q-%d.npy' % seed), diff_qs)
    np.save(os.path.join(log_dir, 'bias_sa-%d.npy' % seed), bias_sas)
    np.save(os.path.join(log_dir, 'bias_max-%d.npy' % seed), bias_maxs)

    return js, diff_qs, bias_sas, bias_maxs


def experiment(seed, log_dir):
    return train_dqn(seed, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-exp", type=int, default=20)
    args = parser.parse_args()

    log_dir = "logs/dqn_car_on_hill"
    os.makedirs(log_dir, exist_ok=True)

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(k, log_dir) for k in range(args.n_exp))

    Js      = np.array([o[0] for o in out])
    Qs      = np.array([o[1] for o in out])
    Bias_sa = np.array([o[2] for o in out])
    Bias_max= np.array([o[3] for o in out])

    np.save(os.path.join(log_dir, 'J.npy'), Js)
    np.save(os.path.join(log_dir, 'Q.npy'), Qs)
    np.save(os.path.join(log_dir, 'bias_sa.npy'), Bias_sa)
    np.save(os.path.join(log_dir, 'bias_max.npy'), Bias_max)

    print('J: ',       np.mean(Js, 0))
    print('Q diff: ',  np.mean(Qs, 0))
