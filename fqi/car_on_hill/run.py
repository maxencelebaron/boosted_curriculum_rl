import argparse
import pathlib
import pickle
import yaml

from joblib import Parallel, delayed
from fqi.fast_extra_tress import FastExtraTreesActionRegressor
from fqi.network_fqi import NeuralRegressor
from tqdm import trange
import numpy as np

from fqi.fqi import BoostedFQI
from fqi.neural_fqi import BoostedNeuralFQI, NeuralFQI
from fqi.car_on_hill.solver import solve_car_on_hill

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter


def experiment(exp_id, ms, boosted, neural, iters_per_env, monitor_loss=False, data_dir="data"):
    seed = 95 + exp_id
    np.random.seed(seed)
    print("Running with seed %d" % seed)

    if neural and boosted:
        alg = BoostedNeuralFQI
    elif neural:
        alg = NeuralFQI
    elif boosted:
        alg = BoostedFQI
    else:
        alg = FQI

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

    # Test Q
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
    epsilon = Parameter(value=1.)
    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    if neural:
        approximator_cls = NeuralRegressor
        approximator_params = dict(
            input_shape=mdps[0].info.observation_space.shape,
            n_actions=mdps[0].info.action_space.n,
            output_shape=(mdps[0].info.action_space.n,),
            n_models=n_tasks if boosted else 1,
            prediction='sum',
        )
        fit_params = dict(
            lr=4e-3,
            n_epochs=3,
            batch_size=128,
            reinit=False,
        )
    else:
        approximator_cls = FastExtraTreesActionRegressor
        approximator_params = dict(
            input_shape=mdps[0].info.observation_space.shape,
            n_actions=mdps[0].info.action_space.n,
            n_models=n_tasks if boosted else 1,
            n_estimators=50,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
            prediction='sum',
        )
        fit_params = {}

    algorithm_params = dict(n_iterations=1)

    # Agent
    agent = alg(
        mdps[0].info,
        pi,
        approximator_cls,
        quiet=True,
        approximator_params=approximator_params,
        fit_params=fit_params,
        **algorithm_params
    )
    # print(f"DEBUG INIT: type(agent.approximator) = {type(agent.approximator).__name__}")
    # if hasattr(agent.approximator, '_models'):
    #     print(f"DEBUG INIT: len(_models) = {len(agent.approximator._models)}")
    #     print(f"DEBUG INIT: type(_models[0]) = {type(agent.approximator._models[0]).__name__}")
    # print(f"approximator module: {type(agent.approximator).__module__}")
    # print(f"approximator MRO: {type(agent.approximator).__mro__}")

    js = list()
    diff_qs = list()
    all_bias_sa = list()
    all_bias_max = list()
    all_depths = list()
    all_losses = list() if (neural and monitor_loss) else None
    all_q_errors = list() if (neural and monitor_loss) else None
    for i, mdp in enumerate(mdps):
        logger.info('TASK: %d\n-------' % i)
        if boosted:
            agent.set_curriculum_idx_and_reset(i)

        # Algorithm
        core = Core(agent, mdp)

        # Dataset collection
        try:
            with open('%s/dataset_%1.3f.pkl' % (data_dir, mdp._m), 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            agent.policy.set_epsilon(epsilon)
            logger.info('Generating dataset for task %d...' % i)
            dataset = core.evaluate(n_episodes=1000)
            with open('%s/dataset_%1.3f.pkl' % (data_dir, mdp._m), 'wb') as f:
                pickle.dump(dataset, f)

        j_task = list()
        diff_q_task = list()
        bias_sa_task = list()
        bias_max_task = list()
        depth_task = list()
        agent.policy.set_epsilon(test_epsilon)
        idx = np.arange(i + 1) if boosted else 0
        # Loop
        for _ in trange(iters_per_env, dynamic_ncols=True, disable=False, leave=False):
            # Train
            if all_q_errors is not None:
                regressor = agent.approximator.model[i if boosted else 0]
                q_epoch = []
                def _make_q_cb(approx, states, actions, q_star, ens_idx):
                    def _cb():
                        qs = approx.predict(states, actions, idx=ens_idx)
                        q_epoch.append(np.linalg.norm(qs - q_star, ord=1) / len(qs))
                    return _cb
                regressor.epoch_callback = _make_q_cb(
                    agent.approximator, test_states, test_actions, test_q[i], idx)

            agent.fit(dataset)

            # reg = agent.approximator.model[0].model[i if boosted else 0]
            # depth_task.append(reg.last_depths)

            if all_losses is not None:
                regressor = agent.approximator.model[i if boosted else 0]
                # print(f"DEBUG: regressor type = {type(regressor)}, has last_loss_history = {hasattr(regressor, 'last_loss_history')}")
                all_losses.append(list(regressor.last_loss_history))

            if all_q_errors is not None:
                regressor.epoch_callback = None
                all_q_errors.append(q_epoch)

            # Test
            test_dataset = core.evaluate(initial_states=test_states, quiet=True)

            j_task.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))
            qs = agent.approximator.predict(test_states, test_actions, idx=idx)
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
        all_depths.append(depth_task)

    model_arch = str(agent.approximator.model[0]._model) if neural else None
    return js, diff_qs, all_bias_sa, all_bias_max, all_losses, all_q_errors, all_depths, fit_params, approximator_params, approximator_cls.__name__, model_arch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--use-curriculum", action='store_true')
    parser.add_argument("--use-boosting", action='store_true')
    parser.add_argument(
        "--use-neural",
        action='store_true',
        help="Use neural network as the weak learner in boosting"
    )
    parser.add_argument("--n-exp", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--monitor-loss", action='store_true')
    args = parser.parse_args()

    if args.use_curriculum:
        ms = [.8, 1., 1.2]
        iters_per_env = 20
    else:
        if args.use_boosting:
            ms = [1.2, 1.2, 1.2]
            iters_per_env = 20
        else:
            ms = [1.2]
            iters_per_env = 60

    print("=========================")

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(
            exp_id,
            ms,
            args.use_boosting,
            args.use_neural,
            iters_per_env,
            args.monitor_loss,
            args.data_dir,
        )
        for exp_id in range(args.n_exp))
    Js = [o[0] for o in out]
    Qs = [o[1] for o in out]
    Bias_sa = [o[2] for o in out]
    Bias_max = [o[3] for o in out]
    Losses = [o[4] for o in out]
    Q_errors = [o[5] for o in out]
    Depths = [o[6] for o in out]
    fit_params = out[0][7]
    approximator_params = out[0][8]
    approximator_cls_name = out[0][9]
    model_arch = out[0][10]

    # Summary folder
    if args.use_neural:
        boost = 'boosted' if args.use_boosting else 'no_boosted'
        cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
        reinit = 'reinit' if fit_params['reinit'] else 'no_reinit'
        folder_name = f'./logs/neural_{boost}_{cur}_lr{fit_params["lr"]}_ep{fit_params["n_epochs"]}_bs{fit_params["batch_size"]}_{reinit}'
    else:
        alg = 'boosted' if args.use_boosting else 'no_boosted'
        cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
        folder_name = './logs/' + alg + '_' + cur
    if args.use_neural:
        print(f"  lr:             {fit_params['lr']}")
        print(f"  n_epochs:       {fit_params['n_epochs']}")
        print(f"  batch_size:     {fit_params['batch_size']}")
        print(f"  reinit:         {fit_params['reinit']}")
        print("=========================")
    print(f"Output folder: {folder_name}")
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    config = {
        'approximator': approximator_cls_name,
        'use_curriculum': args.use_curriculum,
        'use_boosting': args.use_boosting,
        'use_neural': args.use_neural,
        'n_exp': args.n_exp,
        'ms': ms,
        'iters_per_env': iters_per_env,
        'approximator_params': {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in approximator_params.items()
            if k != 'random_state'
        },
        'fit_params': fit_params,
        'model_architecture': model_arch,
    }
    with open(folder_name + '/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    np.save(folder_name + '/J.npy', Js)
    np.save(folder_name + '/Q.npy', Qs)
    np.save(folder_name + '/bias_sa.npy', Bias_sa)
    np.save(folder_name + '/bias_max.npy', Bias_max)
    np.save(folder_name + '/depths.npy', np.array(Depths))
    if args.monitor_loss and args.use_neural:
        np.save(folder_name + '/losses.npy', np.array(Losses, dtype=float))
        np.save(folder_name + '/q_errors_per_epoch.npy', np.array(Q_errors, dtype=float))

    print('J: ', np.mean(Js, 0))
    print('Q diff: ', np.mean(Qs, 0))
