from joblib import Parallel, delayed
from dataclasses import dataclass, field

import os
import yaml
import torch
import pathlib
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import tyro
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


@dataclass
class Args:
    use_curriculum: bool = False
    """if set, train on ms=[0.8, 1.0, 1.2] with curriculum_timesteps per task"""
    use_boosting: bool = False
    """if set, use BoostedCarOnHillDQN"""
    n_jobs: int = 4
    """number of parallel jobs"""
    n_exp: int = 5
    """number of experiments (seeds)"""
    n_eval_points: int = 20
    """number of evaluation checkpoints per task"""
    curriculum_timesteps: tuple[int, int, int] = (30_000, 40_000, 50_000)
    """timesteps per task when use_curriculum=True"""
    transition_steps: int = 1000
    """steps with fixed low epsilon to fill buffer when switching tasks (curriculum only)"""
    transition_eps: float = 0.05
    """starting epsilon for tasks i>0"""
    n_timesteps: int = 120000
    """timesteps for the single task when use_curriculum=False"""
    train_freq: int = 16
    """number of env steps between gradient updates"""
    gradient_steps: int = 8
    """number of gradient updates per training call"""
    exploration_fraction: float = 0.2
    """fraction of task timesteps over which epsilon is annealed"""
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


def experiment(exp_id, ms, timesteps_per_task, args):
    seed = exp_id
    np.random.seed(seed)

    alg = BoostedCarOnHillDQN if args.use_boosting else CarOnHillDQN
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

    optimizer = {'class': optim.Adam, 'params': dict(lr=args.learning_rate)}
    approximator_params = dict(
        network=Network,
        input_shape=mdps[0].info.observation_space.shape,
        output_shape=(mdps[0].info.action_space.n,),
        n_actions=mdps[0].info.action_space.n,
        loss=F.mse_loss,
        optimizer=optimizer,
        use_cuda=True,
        prediction='sum',
    )
    if args.use_boosting:
        approximator_params['n_models'] = n_tasks

    algorithm_params = dict(
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_interval,
        initial_replay_size=args.learning_starts,
        max_replay_size=args.buffer_size,
    )

    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=Parameter(value=1.))

    agent = alg(
        mdps[0].info,
        pi,
        TorchApproximator,
        gradient_steps=args.gradient_steps,
        approximator_params=approximator_params,
        **algorithm_params
    )

    n_states = len(test_states) // 2
    js, diff_qs, bias_sas, bias_maxs = [], [], [], []

    for i, (mdp, n_timesteps_task) in enumerate(zip(mdps, timesteps_per_task)):
        core = Core(agent, mdp)

        if args.use_boosting:
            agent.set_curriculum_idx_and_reset(i)
        elif i > 0:
            agent._replay_memory.reset()
            # fill buffer with learned policy before resuming training
            pi.set_epsilon(Parameter(value=args.transition_eps))
            core.learn(n_steps=args.transition_steps, n_steps_per_fit=args.train_freq)

        steps_per_eval = n_timesteps_task // args.n_eval_points
        if i == 0:
            n_explore = int(n_timesteps_task * args.exploration_fraction)
            epsilon = LinearParameter(value=1.0, threshold_value=args.exploration_final_eps, n=n_explore)
        else:
            epsilon = Parameter(value=0.)

        predict_kwargs = dict(idx=np.arange(i + 1)) if args.use_boosting else {}

        j_task, diff_q_task, bias_sa_task, bias_max_task = [], [], [], []

        for _ in trange(args.n_eval_points, dynamic_ncols=True, leave=False):
            pi.set_epsilon(epsilon)
            core.learn(n_steps=steps_per_eval, n_steps_per_fit=args.train_freq)

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
    args = tyro.cli(Args)

    if args.use_curriculum:
        ms = [.8, 1., 1.2]
        timesteps_per_task = list(args.curriculum_timesteps)
    else:
        ms = [1.2]
        timesteps_per_task = [args.n_timesteps]

    boost = 'boosted' if args.use_boosting else 'no_boosted'
    cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
    log_dir = 'logs/dqn_%s_%s' % (boost, cur)
    os.makedirs(log_dir, exist_ok=True)

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(k, ms, timesteps_per_task, args)
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
        'ms': ms,
        'timesteps_per_task': timesteps_per_task,
        'n_eval_points': args.n_eval_points,
        'n_exp': args.n_exp,
        'n_jobs': args.n_jobs,
        'dqn_params': {
            'train_freq': args.train_freq,
            'gradient_steps': args.gradient_steps,
            'exploration_fraction': args.exploration_fraction,
            'exploration_final_eps': args.exploration_final_eps,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'buffer_size': args.buffer_size,
            'learning_starts': args.learning_starts,
            'target_update_interval': args.target_update_interval,
        },
    }
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print('J: ', np.mean(Js, 0))
    print('Q diff: ', np.mean(Qs, 0))
