from joblib import Parallel, delayed

import os
import yaml
import torch
import pathlib
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.environments.car_on_hill import CarOnHill

from car_on_hill_dqn.dqn import CarOnHillDQN, BoostedCarOnHillDQN
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


def experiment(exp_id, ms, use_boosting, n_eval_points=N_EVAL_POINTS):
    seed = exp_id
    np.random.seed(seed)

    alg = BoostedCarOnHillDQN if use_boosting else CarOnHillDQN
    n_tasks = len(ms)

    mdps = [CarOnHill() for _ in range(n_tasks)]
    for m, mdp in zip(ms, mdps):
        mdp._m = m
    names = ['%1.3f' % mdp._m for mdp in mdps]

    # Test grid (same as fqi/car_on_hill/run.py)
    test_states_0 = np.linspace(
        mdps[0].info.observation_space.low[0],
        mdps[0].info.observation_space.high[0], 10)
    test_states_1 = np.linspace(
        mdps[0].info.observation_space.low[1],
        mdps[0].info.observation_space.high[1], 10)
    test_states = []
    for s0 in test_states_0:
        for s1 in test_states_1:
            test_states += [s0, s1]
    test_states = np.array([test_states]).repeat(2, 0).reshape(-1, 2)
    test_actions = np.array(
        [np.zeros(len(test_states) // 2),
         np.ones(len(test_states) // 2)]).reshape(-1, 1).astype(int)

    # Q* per task (cached)
    test_q = []
    for name, mdp in zip(names, mdps):
        q_path = 'data/test_q_%s.npy' % name
        try:
            test_q.append(np.load(q_path))
        except FileNotFoundError:
            pathlib.Path('data').mkdir(parents=True, exist_ok=True)
            q_vals = np.array(solve_car_on_hill(mdp, test_states, test_actions, mdp.info.gamma))
            np.save(q_path, q_vals)
            test_q.append(q_vals)
    test_q = np.array(test_q)

    optimizer = {'class': optim.Adam, 'params': dict(lr=LEARNING_RATE)}
    approximator_params = dict(
        network=Network,
        input_shape=mdps[0].info.observation_space.shape,
        output_shape=(mdps[0].info.action_space.n,),
        n_actions=mdps[0].info.action_space.n,
        loss=F.mse_loss,
        optimizer=optimizer,
        use_cuda=True,
    )
    if use_boosting:
        approximator_params['n_models'] = n_tasks

    algorithm_params = dict(
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_INTERVAL,
        initial_replay_size=LEARNING_STARTS,
        max_replay_size=BUFFER_SIZE,
    )

    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=Parameter(value=1.))

    agent = alg(mdps[0].info, pi, TorchApproximator,
                gradient_steps=GRADIENT_STEPS,
                approximator_params=approximator_params,
                **algorithm_params)

    steps_per_eval = N_TIMESTEPS // n_eval_points
    n_explore = int(N_TIMESTEPS * EXPLORATION_FRACTION)
    n_states = len(test_states) // 2

    js, diff_qs, bias_sas, bias_maxs = [], [], [], []

    for i, mdp in enumerate(mdps):
        if use_boosting:
            agent.set_curriculum_idx_and_reset(i)

        epsilon = LinearParameter(value=1.0, threshold_value=EXPLORATION_FINAL_EPS, n=n_explore)
        core = Core(agent, mdp)
        predict_kwargs = dict(idx=np.arange(i + 1)) if use_boosting else {}

        j_task, diff_q_task, bias_sa_task, bias_max_task = [], [], [], []

        for _ in trange(n_eval_points, dynamic_ncols=True, leave=False):
            pi.set_epsilon(epsilon)
            core.learn(n_steps=steps_per_eval, n_steps_per_fit=TRAIN_FREQ)

            pi.set_epsilon(test_epsilon)
            test_dataset = core.evaluate(initial_states=test_states, quiet=True)

            j_task.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))
            qs = agent.approximator.predict(test_states, test_actions, **predict_kwargs)
            diff_q_task.append(np.linalg.norm(qs - test_q[i], ord=1) / len(qs))
            bias_sa_task.append(np.mean(qs - test_q[i]))
            q_hat_max = np.maximum(qs[:n_states], qs[n_states:])
            q_star_max = np.maximum(test_q[i][:n_states], test_q[i][n_states:])
            bias_max_task.append(np.mean(q_hat_max - q_star_max))

        js.append(j_task)
        diff_qs.append(diff_q_task)
        bias_sas.append(bias_sa_task)
        bias_maxs.append(bias_max_task)

    return js, diff_qs, bias_sas, bias_maxs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-curriculum", action='store_true')
    parser.add_argument("--use-boosting", action='store_true')
    parser.add_argument("--n-exp", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    if args.use_curriculum:
        ms = [.8, 1., 1.2]
    else:
        ms = [1.2, 1.2, 1.2] if args.use_boosting else [1.2]

    n_eval_points = N_EVAL_POINTS

    boost = 'boosted' if args.use_boosting else 'no_boosted'
    cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
    log_dir = 'logs/dqn_%s_%s' % (boost, cur)
    os.makedirs(log_dir, exist_ok=True)

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(k, ms, args.use_boosting, n_eval_points)
        for k in range(args.n_exp))

    Js = np.array([o[0] for o in out])
    Qs = np.array([o[1] for o in out])
    Bias_sa = np.array([o[2] for o in out])
    Bias_max = np.array([o[3] for o in out])

    np.save(os.path.join(log_dir, 'J.npy'), Js)
    np.save(os.path.join(log_dir, 'Q.npy'), Qs)
    np.save(os.path.join(log_dir, 'bias_sa.npy'), Bias_sa)
    np.save(os.path.join(log_dir, 'bias_max.npy'), Bias_max)

    config = {
        'algorithm': 'DQN',
        'use_curriculum': args.use_curriculum,
        'use_boosting': args.use_boosting,
        'n_exp': args.n_exp,
        'n_jobs': args.n_jobs,
        'ms': ms,
        'n_eval_points': n_eval_points,
        'dqn_params': {
            'n_timesteps': N_TIMESTEPS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'buffer_size': BUFFER_SIZE,
            'learning_starts': LEARNING_STARTS,
            'target_update_interval': TARGET_UPDATE_INTERVAL,
            'train_freq': TRAIN_FREQ,
            'gradient_steps': GRADIENT_STEPS,
            'exploration_fraction': EXPLORATION_FRACTION,
            'exploration_final_eps': EXPLORATION_FINAL_EPS,
        },
    }
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print('J: ', np.mean(Js, 0))
    print('Q diff: ', np.mean(Qs, 0))
