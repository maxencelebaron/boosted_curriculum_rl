import argparse
import pathlib
import yaml

from joblib import Parallel, delayed
from tqdm import trange
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from fqi.network_fqi import DQNNetwork
from fqi.car_on_hill.solver import solve_car_on_hill
from linear_dqn.dqn import BoostedDQN

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter


def experiment(exp_id, ms, boosted, dqn_params, n_eval_points=20, data_dir="data"):
    seed = 95 + exp_id
    np.random.seed(seed)
    print("Running with seed %d" % seed)

    alg = BoostedDQN if boosted else DQN

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment algorithm: ' + alg.__name__)

    # MDP
    mdps = [CarOnHill() for _ in range(len(ms))]
    for m, mdp in zip(ms, mdps):
        mdp._m = m
    n_tasks = len(mdps)

    names = ['%1.3f' % (mdp._m) for mdp in mdps]

    test_states_0 = np.linspace(
        mdps[0].info.observation_space.low[0],
        mdps[0].info.observation_space.high[0],
        10
    )
    test_states_1 = np.linspace(
        mdps[0].info.observation_space.low[1],
        mdps[0].info.observation_space.high[1],
        10
    )
    test_states = list()
    for s0 in test_states_0:
        for s1 in test_states_1:
            test_states += [s0, s1]
    test_states = np.array([test_states]).repeat(2, 0).reshape(-1, 2)
    test_actions = np.array(
        [np.zeros(len(test_states) // 2),
         np.ones(len(test_states) // 2)]).reshape(-1, 1).astype(int)

    # Test Q*
    test_q = list()
    for i, mdp in enumerate(mdps):
        try:
            test_q.append(np.load('data/test_q_%s.npy' % names[i]).tolist())
        except FileNotFoundError:
            logger.info('Generating test Q-values for task %d...' % i)
            current_test_q = solve_car_on_hill(mdp, test_states, test_actions, mdp.info.gamma)
            pathlib.Path('data').mkdir(parents=True, exist_ok=True)
            np.save('data/test_q_%s.npy' % names[i], current_test_q)
            test_q.append(current_test_q)
    test_q = np.array(test_q)

    # Policy
    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=Parameter(value=1.))

    # Approximator
    n_timesteps = dqn_params['n_timesteps']
    optimizer = {'class': optim.Adam, 'params': {'lr': dqn_params['learning_rate']}}
    approximator_params = dict(
        network=DQNNetwork,
        input_shape=mdps[0].info.observation_space.shape,
        output_shape=(mdps[0].info.action_space.n,),
        n_actions=mdps[0].info.action_space.n,
        optimizer=optimizer,
        loss=F.mse_loss,
        use_cuda=True,
    )
    if boosted:
        approximator_params['n_models'] = n_tasks

    algorithm_params = dict(
        batch_size=dqn_params['batch_size'],
        target_update_frequency=dqn_params['target_update_interval'],
        initial_replay_size=dqn_params['learning_starts'],
        max_replay_size=dqn_params['buffer_size'],
    )

    agent = alg(mdps[0].info, pi, TorchApproximator,
                approximator_params=approximator_params,
                **algorithm_params)

    n_steps_per_fit = max(1, dqn_params['train_freq'] // dqn_params['gradient_steps'])
    steps_per_eval = n_timesteps // n_eval_points

    js = []
    diff_qs = []
    all_bias_sa = []
    all_bias_max = []

    for i, mdp in enumerate(mdps):
        logger.info('TASK: %d\n-------' % i)
        if boosted:
            agent.set_curriculum_idx_and_reset(i)

        # Fresh epsilon schedule per task
        n_explore = int(n_timesteps * dqn_params['exploration_fraction'])
        epsilon = LinearParameter(
            value=1.0,
            threshold_value=dqn_params['exploration_final_eps'],
            n=n_explore,
        )
        pi.set_epsilon(epsilon)

        core = Core(agent, mdp)
        predict_kwargs = dict(idx=np.arange(i + 1)) if boosted else {}

        j_task, diff_q_task, bias_sa_task, bias_max_task = [], [], [], []

        for _ in trange(n_eval_points, dynamic_ncols=True, leave=False):
            pi.set_epsilon(epsilon)
            core.learn(n_steps=steps_per_eval, n_steps_per_fit=n_steps_per_fit)

            pi.set_epsilon(test_epsilon)
            test_dataset = core.evaluate(initial_states=test_states, quiet=True)

            j_task.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))
            qs = agent.approximator.predict(test_states, test_actions, **predict_kwargs)
            diff_q_task.append(np.linalg.norm(qs - test_q[i], ord=1) / len(qs))

            n_states = len(qs) // 2
            bias_sa_task.append(np.mean(qs - test_q[i]))
            q_hat_max = np.maximum(qs[:n_states], qs[n_states:])
            q_star_max = np.maximum(test_q[i][:n_states], test_q[i][n_states:])
            bias_max_task.append(np.mean(q_hat_max - q_star_max))

        js.append(j_task)
        diff_qs.append(diff_q_task)
        all_bias_sa.append(bias_sa_task)
        all_bias_max.append(bias_max_task)

    return js, diff_qs, all_bias_sa, all_bias_max, dqn_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--use-curriculum", action='store_true')
    parser.add_argument("--use-boosting", action='store_true')
    parser.add_argument("--n-exp", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=10)

    # DQN hyperparameters
    parser.add_argument("--n-timesteps", type=float, default=1.2e5)
    parser.add_argument("--learning-rate", type=float, default=4e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=600)
    parser.add_argument("--train-freq", type=int, default=16)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-final-eps", type=float, default=0.07)
    args = parser.parse_args()

    dqn_params = dict(
        n_timesteps=int(args.n_timesteps),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
    )

    if args.use_curriculum:
        ms = [.8, 1., 1.2]
    else:
        ms = [1.2, 1.2, 1.2] if args.use_boosting else [1.2]

    n_eval_points = 20

    print("=========================")

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(
            exp_id,
            ms,
            args.use_boosting,
            dqn_params,
            n_eval_points,
            args.data_dir,
        )
        for exp_id in range(args.n_exp))

    Js = [o[0] for o in out]
    Qs = [o[1] for o in out]
    Bias_sa = [o[2] for o in out]
    Bias_max = [o[3] for o in out]
    dqn_params = out[0][4]

    boost = 'boosted' if args.use_boosting else 'no_boosted'
    cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
    folder_name = f'./logs/dqn_{boost}_{cur}_lr{dqn_params["learning_rate"]}_bs{dqn_params["batch_size"]}'

    print(f"Output folder: {folder_name}")
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    config = {
        'algorithm': 'DQN',
        'use_curriculum': args.use_curriculum,
        'use_boosting': args.use_boosting,
        'n_exp': args.n_exp,
        'n_jobs': args.n_jobs,
        'ms': ms,
        'n_eval_points': n_eval_points,
        'data_dir': args.data_dir,
        'dqn_params': {
            'n_timesteps': dqn_params['n_timesteps'],
            'learning_rate': dqn_params['learning_rate'],
            'batch_size': dqn_params['batch_size'],
            'buffer_size': dqn_params['buffer_size'],
            'learning_starts': dqn_params['learning_starts'],
            'target_update_interval': dqn_params['target_update_interval'],
            'train_freq': dqn_params['train_freq'],
            'gradient_steps': dqn_params['gradient_steps'],
            'exploration_fraction': dqn_params['exploration_fraction'],
            'exploration_final_eps': dqn_params['exploration_final_eps'],
        },
    }
    with open(folder_name + '/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    np.save(folder_name + '/J.npy', Js)
    np.save(folder_name + '/Q.npy', Qs)
    np.save(folder_name + '/bias_sa.npy', Bias_sa)
    np.save(folder_name + '/bias_max.npy', Bias_max)

    print('J: ', np.mean(Js, 0))
    print('Q diff: ', np.mean(Qs, 0))
