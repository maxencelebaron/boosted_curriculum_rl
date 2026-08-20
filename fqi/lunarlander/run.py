"""Online FQI experiments on discrete LunarLander."""

import argparse
import pathlib
import pickle

import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import trange

from fqi.fast_extra_tress import FastExtraTreesActionRegressor
from fqi.fqi import BoostedFQI
from fqi.network_fqi import NeuralRegressor
from fqi.neural_fqi import BoostedNeuralFQI, NeuralFQI
from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

from fqi.lunarlander.env import LunarLander


def experiment(
    exp_id,
    wind_powers,
    gravity,
    boosted,
    neural,
    iters_per_env,
    collect_epsilon=0.05,
    n_episodes_collect=1000,
    n_episodes_test=20,
    monitor_loss=False,
    data_dir="data",
):
    seed = 95 + exp_id
    np.random.seed(seed)

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
    logger.info("Experiment algorithm: " + alg.__name__)

    # MDP
    mdps = [
        LunarLander(
            gravity=gravity,
            enable_wind=wind_power > 0,
            wind_power=wind_power,
            turbulence_power=1.5,
        )
        for wind_power in wind_powers
    ]
    n_tasks = len(mdps)

    test_epsilon = Parameter(value=0.0)

    policy = EpsGreedy(epsilon=Parameter(value=1.0))

    if neural:
        approximator_cls = NeuralRegressor
        approximator_params = dict(
            input_shape=mdps[0].info.observation_space.shape,
            n_actions=mdps[0].info.action_space.n,
            output_shape=(mdps[0].info.action_space.n,),
            n_models=n_tasks if boosted else 1,
            prediction="sum",
        )
        fit_params = dict(lr=5e-4, n_epochs=3, batch_size=64, reinit=False)
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
            prediction="sum",
        )
        fit_params = {}

    agent = alg(
        mdps[0].info,
        policy,
        approximator_cls,
        n_iterations=1,
        quiet=True,
        approximator_params=approximator_params,
        fit_params=fit_params,
    )

    returns = []
    all_depths = list()
    losses = [] if neural and monitor_loss else None
    previous_task_trained = False

    for task_idx, (wind_power, mdp) in enumerate(zip(wind_powers, mdps)):
        logger.info("TASK: %d (wind_power=%g)\n-------" % (task_idx, wind_power))
        dataset_path = pathlib.Path(data_dir) / (
            "dataset_task_%d_wind_power_%g_seed_%d.pkl" % (task_idx, wind_power, seed)
        )
        try:
            with dataset_path.open("rb") as stream:
                dataset = pickle.load(stream)
        except FileNotFoundError:
            collection_epsilon = collect_epsilon if previous_task_trained else 1.0
            agent.policy.set_epsilon(Parameter(value=collection_epsilon))
            logger.info(
                "Collecting %d episodes with epsilon=%g..."
                % (n_episodes_collect, collection_epsilon)
            )
            dataset = Core(agent, mdp).evaluate(
                n_episodes=n_episodes_collect
            )
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_path.open("wb") as stream:
                pickle.dump(dataset, stream)

        if boosted:
            agent.set_curriculum_idx_and_reset(task_idx)

        core = Core(agent, mdp)
        agent.policy.set_epsilon(test_epsilon)
        task_returns = []
        depth_task = []

        for _ in trange(iters_per_env, dynamic_ncols=True, leave=False):
            agent.fit(dataset)
            if not neural:
                regressor = agent.approximator.model[0].model[
                    task_idx if boosted else 0
                ]
                depth_task.append(regressor.last_depths)
            if losses is not None:
                regressor = agent.approximator.model[task_idx if boosted else 0]
                losses.append(list(regressor.last_loss_history))
            test_dataset = core.evaluate(n_episodes=n_episodes_test, quiet=True)
            task_returns.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))

        returns.append(task_returns)
        all_depths.append(depth_task)
        previous_task_trained = True
        mdp.stop()

    model_arch = str(agent.approximator.model[0]._model) if neural else None
    return returns, losses, fit_params, approximator_params, approximator_cls.__name__, model_arch, all_depths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_online")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--use-boosting", action="store_true")
    parser.add_argument("--use-neural", action="store_true")
    parser.add_argument("--monitor-loss", action="store_true")
    parser.add_argument("--wind-powers", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0])
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--iterations-per-task", type=int, default=20)
    parser.add_argument("--n-exp", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--collect-epsilon", type=float, default=0.05)
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--n-test-episodes", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_curriculum:
        wind_powers = args.wind_powers
        iterations_per_task = args.iterations_per_task
    elif args.use_boosting:
        wind_powers = [args.wind_powers[-1]] * len(args.wind_powers)
        iterations_per_task = args.iterations_per_task
    else:
        wind_powers = [args.wind_powers[-1]] * len(args.wind_powers)
        iterations_per_task = args.iterations_per_task

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(
            exp_id,
            wind_powers,
            args.gravity,
            args.use_boosting,
            args.use_neural,
            iterations_per_task,
            args.collect_epsilon,
            args.n_episodes,
            args.n_test_episodes,
            args.monitor_loss,
            args.data_dir,
        )
        for exp_id in range(args.n_exp)
    )

    returns = [result[0] for result in out]
    losses = [result[1] for result in out]
    depths = [result[6] for result in out]
    fit_params, approximator_params = out[0][2], out[0][3]
    boost = "boosted" if args.use_boosting else "no_boosted"
    curriculum = "curriculum" if args.use_curriculum else "no_curriculum"
    learner = "neural" if args.use_neural else "trees"
    output_dir = pathlib.Path(args.output_dir) / f"{learner}_{boost}_{curriculum}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = dict(
        approximator=out[0][4],
        use_curriculum=args.use_curriculum,
        use_boosting=args.use_boosting,
        use_neural=args.use_neural,
        n_exp=args.n_exp,
        gravity=args.gravity,
        wind_powers=wind_powers,
        iterations_per_task=iterations_per_task,
        collect_epsilon=args.collect_epsilon,
        n_episodes_collect=args.n_episodes,
        n_episodes_test=args.n_test_episodes,
        approximator_params={
            key: list(value) if isinstance(value, tuple) else value
            for key, value in approximator_params.items()
            if key != "random_state"
        },
        fit_params=fit_params,
        model_architecture=out[0][5],
    )
    with (output_dir / "config.yaml").open("w") as stream:
        yaml.dump(config, stream, default_flow_style=False, sort_keys=False)
    np.save(output_dir / "J.npy", np.asarray(returns))
    if not args.use_neural:
        np.save(output_dir / "depths.npy", np.asarray(depths))
    if args.monitor_loss and args.use_neural:
        np.save(output_dir / "losses.npy", np.asarray(losses, dtype=float))

    print("Output folder:", output_dir)
    print("J:", np.mean(returns, axis=0))
