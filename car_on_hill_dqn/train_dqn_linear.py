from joblib import Parallel, delayed
from dataclasses import dataclass

import os
import torch
import pathlib
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import tyro
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


@dataclass
class Args:
    n_jobs: int = 4
    """number of parallel jobs"""
    n_exp: int = 4
    """number of experiments (seeds)"""
    n_timesteps: int = 120000
    """total environment steps per experiment"""
    n_eval_points: int = 60
    """number of evaluation checkpoints"""
    train_freq: int = 16
    """number of env steps between gradient updates"""
    gradient_steps: int = 8
    """number of gradient updates per training call"""
    exploration_fraction: float = 0.2
    """fraction of n_timesteps over which epsilon is annealed"""
    exploration_final_eps: float = 0.07
    """final value of epsilon"""
    learning_rate: float = 4e-3
    """learning rate of the Adam optimizer"""
    batch_size: int = 128
    """batch size sampled from the replay buffer"""
    buffer_size: int = 10000
    """maximum replay buffer size"""
    learning_starts: int = 1000
    """number of steps before learning starts"""
    target_update_interval: int = 600
    """number of gradient steps between target network updates"""


def train_dqn(seed, log_dir, args):
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

    optimizer = {'class': optim.Adam, 'params': dict(lr=args.learning_rate)}

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
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_interval,
        initial_replay_size=args.learning_starts,
        max_replay_size=args.buffer_size,
    )

    steps_per_eval = args.n_timesteps // args.n_eval_points
    n_explore = int(args.n_timesteps * args.exploration_fraction)

    epsilon = LinearParameter(value=1.0, threshold_value=args.exploration_final_eps, n=n_explore)
    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon)

    agent = CarOnHillDQN(
        mdp.info,
        pi,
        TorchApproximator,
        gradient_steps=args.gradient_steps,
        approximator_params=approximator_params,
        **algorithm_params
    )
    core = Core(agent, mdp)

    js, diff_qs, bias_sas, bias_maxs = [], [], [], []
    n_states = len(test_states) // 2

    for _ in range(args.n_eval_points):
        pi.set_epsilon(epsilon)
        core.learn(n_steps=steps_per_eval, n_steps_per_fit=args.train_freq)

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


def experiment(seed, log_dir, args):
    return train_dqn(seed, log_dir, args)


if __name__ == "__main__":
    args = tyro.cli(Args)

    log_dir = "logs/dqn_car_on_hill"
    os.makedirs(log_dir, exist_ok=True)

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(k, log_dir, args) for k in range(args.n_exp))

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
