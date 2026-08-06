from __future__ import annotations

import importlib
import tyro
import pathlib
import pickle
import yaml
from dataclasses import dataclass, asdict

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import trange

from fqi.neural_fqi import BoostedNeuralFQI, NeuralFQI
from fqi.car_on_hill.solver import solve_car_on_hill
from fqi.utils.growing_network import pre_growth_optimize, compute_metrics

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.parameters import Parameter


# Maps growth_mode → dotted module path
GROWTH_MODULES = {
    "random":          "fqi.2_network_fqi_grow_randomly",
    "svd":             "fqi.3_network_fqi_optimizer_plus_svd",
    "gromo_one_layer": "fqi.4_network_fqi_tiny_one_layer",
}


@dataclass
class Args:
    # Experiment
    n_exp: int = 10
    """number of independent experiment seeds"""
    n_jobs: int = 1
    """number of parallel jobs (joblib). Keep at 1 when using GPU."""
    use_curriculum: bool = False
    """if set, trains sequentially on ms=[0.8, 1.0, 1.2]"""
    use_boosting: bool = False
    """if set, uses BoostedNeuralFQI (curriculum boosting)"""
    monitor_loss: bool = False
    """track per-epoch loss and Q-error"""
    growth_mode: str = "svd"
    """growth strategy: random | svd | gromo_one_layer """

    # FQI training
    iters_per_env: int = 20
    """FQI iterations per environment"""
    lr: float = 1e-3
    """learning rate"""
    n_epochs: int = 20
    """training epochs per FQI iteration"""
    batch_size: int = 32
    """mini-batch size for FQI training"""

    feature_rank_n_states: int = 2_000
    """number of states to collect for periodic feature rank monitoring"""

    # Network growth
    initial_hidden: int = 128
    """initial encoder hidden size"""
    final_hidden: int = 256
    """target encoder hidden size after all growth events"""
    grow_every: int = 20
    """FQI iterations between growth events (no-curriculum only)"""
    pre_growth_steps: int = 10
    """gradient steps to update current weights before growing"""
    grow_batch_size: int = 1000
    """batch size for the growth computation"""
    bellman_residual_threshold: float = 0.0
    """skip growth if pre-growth final loss is below this"""
    numerical_threshold: float = 1e-6
    """threshold for near-zero singular values"""
    statistical_threshold: float = 0.0
    """gromo sub-selection threshold"""


def _ms_for_args(args: Args) -> list[float]:
    if args.use_curriculum:
        return [0.8, 1.0, 1.2]
    if args.use_boosting:
        return [1.2, 1.2, 1.2]
    return [1.2]


def _compute_grow_batch(
    dataset,
    regressor,
    gamma: float,
    grow_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a mini-batch for the growth step.

    agent.fit() computes TD targets internally without exposing them.
    This function recomputes them on a random subset of the dataset
    so they can be passed to the grow functions (grow_network_*).

    Steps:
      1. Parse the full dataset -> (s, a, r, s', absorbing)
      2. Randomly subsample grow_batch_size transitions
      3. Compute td = r + gamma * max_a Q(s'; theta) with the current network

    Returns (states, actions, td_targets) as PyTorch tensors.
    """
    states, actions, rewards, next_states, absorbing, _ = parse_dataset(dataset)
    n = len(states)
    idx = np.random.choice(n, min(grow_batch_size, n), replace=False)

    s = states[idx]
    a = actions[idx].reshape(-1).astype(int)
    r = rewards[idx]
    s_next = next_states[idx]
    done = absorbing[idx]

    q_next = regressor.predict(s_next)
    q_max = q_next.max(axis=1)
    td = r + gamma * (1 - done) * q_max

    device = next(regressor._model.parameters()).device
    # All tensors are built from numpy → requires_grad=False by construction.
    # The no_grad wrapper in the recomputation block after pre_growth_optimize

    return (
        torch.FloatTensor(s).to(device),
        torch.LongTensor(a).to(device),
        torch.FloatTensor(td).to(device),
        torch.FloatTensor(s_next).to(device),
        torch.FloatTensor(r).to(device),
        torch.FloatTensor(done).to(device),
    )


def _grow_step(
    growth_mode: str,
    module,
    regressor,
    states: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    args: Args,
    neurons_per_step: int,
) -> dict:
    """
    Optionally update the current weights, then grow according to growth_mode.
    Resets the optimizer after any structural change.

    neurons_per_step = (final_hidden - initial_hidden) // n_growth_events

    Returns a dict with keys: grew, neurons_added, pre_growth_losses.
    """
    q_network = regressor._model
    hidden_size_before = regressor._model.encoder_size
    pre_growth_losses = []

    if args.pre_growth_steps > 0:
        if regressor._optimizer is None:
            regressor._optimizer = torch.optim.Adam(
                q_network.parameters(), lr=args.lr
            )
        initial_loss, final_loss, pre_growth_losses = pre_growth_optimize(
            q_network,
            states,
            actions,
            td_targets,
            regressor._optimizer,
            args.pre_growth_steps,
        )
        if final_loss < args.bellman_residual_threshold:
            return {
                "grew": False,
                "neurons_added": 0,
                "pre_growth_losses": pre_growth_losses
            }
        # Recompute TD targets with the updated network weights before growing
        with torch.no_grad():
            q_next_fresh = q_network(next_states).max(dim=1)[0]
            td_targets = rewards + gamma * (1 - done) * q_next_fresh

    if growth_mode == "random":
        new_h = q_network.encoder[0].out_features + neurons_per_step
        new_net = module.grow_network(q_network, new_h)
        regressor._model = new_net
        regressor._optimizer = None

    elif growth_mode == "svd":
        new_net, svd_singular_values = module.grow_network_svd(
            q_network,
            states,
            actions,
            td_targets,
            d_a=neurons_per_step,
            numerical_threshold=args.numerical_threshold,
        )
        regressor._model = new_net
        regressor._optimizer = None

    elif growth_mode == "gromo_one_layer":
        module.grow_network_gromo(
            q_network, states, actions, td_targets,
            maximum_added_neurons=neurons_per_step,
            numerical_threshold=args.numerical_threshold,
            statistical_threshold=args.statistical_threshold,
        )
        regressor._optimizer = None

    hidden_size_after = regressor._model.encoder_size
    neurons_added = hidden_size_after - hidden_size_before
    return {"grew": neurons_added > 0, "neurons_added": neurons_added, "pre_growth_losses": pre_growth_losses}


def experiment(exp_id: int, args: Args) -> tuple:
    seed = 95 + exp_id
    np.random.seed(seed)
    print(f"Running exp {exp_id} (seed={seed}, growth_mode={args.growth_mode})")

    module = importlib.import_module(GROWTH_MODULES[args.growth_mode])

    ms = _ms_for_args(args)
    n_growth_events = (
        len(ms) - 1 if len(ms) > 1
        else (args.iters_per_env - 1) // args.grow_every
    )
    neurons_per_step = (args.final_hidden - args.initial_hidden) // n_growth_events
    alg = BoostedNeuralFQI if args.use_boosting else NeuralFQI

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info(f'Experiment: {alg.__name__} | growth: {args.growth_mode}')

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
    test_states = []
    for s0 in test_states_0:
        for s1 in test_states_1:
            test_states += [s0, s1]
    test_states = np.array([test_states]).repeat(2, 0).reshape(-1, 2)
    test_actions = np.array(
        [np.zeros(len(test_states) // 2),
         np.ones(len(test_states) // 2)]
    ).reshape(-1, 1).astype(int)

    # Reference Q-values
    test_q = []
    for i, mdp in enumerate(mdps):
        try:
            test_q.append(np.load('data/test_q_%s.npy' % names[i]).tolist())
        except FileNotFoundError:
            logger.info('Generating test Q-values for task %d...' % i)
            current_test_q = solve_car_on_hill(
                mdp, test_states, test_actions, mdp.info.gamma
            )
            pathlib.Path('data').mkdir(parents=True, exist_ok=True)
            np.save('data/test_q_%s.npy' % names[i], current_test_q)
            test_q.append(current_test_q)
    test_q = np.array(test_q)

    epsilon = Parameter(value=1.)
    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    approximator_cls = module.NeuralRegressor
    approximator_params = dict(
        input_shape=mdps[0].info.observation_space.shape,
        n_actions=mdps[0].info.action_space.n,
        output_shape=(mdps[0].info.action_space.n,),
        n_models=n_tasks if args.use_boosting else 1,
        prediction='sum',
        hidden_size=args.initial_hidden,
    )
    fit_params = dict(
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        reinit=False,
    )

    agent = alg(
        mdps[0].info,
        pi,
        approximator_cls,
        quiet=True,
        approximator_params=approximator_params,
        fit_params=fit_params,
        n_iterations=1,
    )

    growth_freq = args.grow_every

    js, diff_qs, bias_sas, bias_maxs = [], [], [], []
    all_losses = [] if args.monitor_loss else None
    all_pre_growth_losses = [] if args.monitor_loss else None
    all_q_errors = [] if args.monitor_loss else None
    all_metrics = []
    prev_dataset = None
    feature_split = 0
    pending_growth_info = None

    for i, mdp in enumerate(mdps):
        logger.info('TASK: %d\n-------' % i)
        if args.use_boosting:
            agent.set_curriculum_idx_and_reset(i)

        core = Core(agent, mdp)
        gamma = mdp.info.gamma
        monitoring_data = None

        try:
            with open('data/dataset_%1.3f.pkl' % mdp._m, 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            agent.policy.set_epsilon(epsilon)
            logger.info('Generating dataset for task %d...' % i)
            dataset = core.evaluate(n_episodes=1000)
            with open('data/dataset_%1.3f.pkl' % mdp._m, 'wb') as f:
                pickle.dump(dataset, f)

        # Grow network at task transition using the new task's dataset
        if i > 0:
            regressor = agent.approximator.model[i if args.use_boosting else 0]
            states_g, actions_g, td_g, next_states_g, rewards_g, done_g = _compute_grow_batch(
                dataset,
                regressor,
                gamma,
                args.grow_batch_size,
            )
            feature_split = regressor._model.encoder_size
            pending_growth_info = _grow_step(
                args.growth_mode,
                module,
                regressor,
                states_g,
                actions_g,
                td_g,
                next_states_g,
                rewards_g,
                done_g,
                gamma,
                args,
                neurons_per_step,
            )

        j_task, diff_q_task, bias_sa_task, bias_max_task = [], [], [], []
        agent.policy.set_epsilon(test_epsilon)
        ens_idx = np.arange(i + 1) if args.use_boosting else 0

        for it in trange(
            args.iters_per_env,
            dynamic_ncols=True,
            disable=False,
            leave=False
        ):
            regressor = agent.approximator.model[i if args.use_boosting else 0]

            if args.monitor_loss:
                q_epoch = []

                def _make_q_cb(approx, s, a, q_star, idx):
                    def _cb():
                        qs = approx.predict(s, a, idx=idx)
                        q_epoch.append(
                            np.linalg.norm(qs - q_star, ord=1) / len(qs)
                        )
                    return _cb

                regressor.epoch_callback = _make_q_cb(
                    agent.approximator, test_states, test_actions,
                    test_q[i], ens_idx,
                )

            agent.fit(dataset)

            if args.monitor_loss:
                all_losses.append(list(regressor.last_loss_history))
                regressor.epoch_callback = None
                all_q_errors.append(q_epoch)

            # Fix monitoring batch on first call per task, then compute metrics
            if monitoring_data is None:
                states_all, actions_all, rewards_all, next_states_all, absorbing_all, _ = parse_dataset(dataset)
                n_mon = min(args.feature_rank_n_states, len(states_all))
                monitoring_data = dict(
                    states=torch.FloatTensor(states_all[:n_mon]),
                    actions=torch.LongTensor(actions_all[:n_mon].reshape(-1)),
                    next_states=next_states_all[:n_mon],
                    rewards=rewards_all[:n_mon],
                    absorbing=absorbing_all[:n_mon],
                )
            q_next_mon = regressor.predict(monitoring_data['next_states']).max(axis=1)
            td_mon = (
                monitoring_data['rewards']
                + gamma * (1 - monitoring_data['absorbing']) * q_next_mon
            )
            metrics = compute_metrics(
                model=regressor._model,
                monitoring_states=monitoring_data['states'],
                monitoring_actions=monitoring_data['actions'],
                monitoring_targets=torch.FloatTensor(td_mon),
                feature_split=feature_split,
            )

            metrics["hidden_size"] = regressor._model.encoder_size

            # Consume task-boundary growth info (attach to first available iteration)
            if pending_growth_info is not None:
                if pending_growth_info["grew"]:
                    metrics["grew"] = True
                    metrics["neurons_added"] = pending_growth_info["neurons_added"]
                    if args.monitor_loss:
                        all_pre_growth_losses.append(pending_growth_info["pre_growth_losses"])
                pending_growth_info = None

            if len(ms) == 1 and (it + 1) % growth_freq == 0 and (it + 1) < args.iters_per_env:
                states_g, actions_g, td_g, next_states_g, rewards_g, done_g = _compute_grow_batch(
                    dataset,
                    regressor,
                    gamma,
                    args.grow_batch_size,
                )
                feature_split = regressor._model.encoder_size
                growth_info = _grow_step(
                    args.growth_mode,
                    module,
                    regressor,
                    states_g,
                    actions_g,
                    td_g,
                    next_states_g,
                    rewards_g,
                    done_g,
                    gamma,
                    args,
                    neurons_per_step,
                )
                if growth_info["grew"]:
                    metrics["grew"] = True
                    metrics["neurons_added"] = growth_info["neurons_added"]
                    if args.monitor_loss:
                        all_pre_growth_losses.append(growth_info["pre_growth_losses"])

            all_metrics.append(metrics)

            test_dataset = core.evaluate(initial_states=test_states, quiet=True)
            j_task.append(np.mean(compute_J(test_dataset, gamma)))
            qs = agent.approximator.predict(test_states, test_actions, idx=ens_idx)
            diff_q_task.append(np.linalg.norm(qs - test_q[i], ord=1) / len(qs))

            n_states = len(qs) // 2
            bias_sa_task.append(np.mean(qs - test_q[i]))
            q_hat_max = np.maximum(qs[:n_states], qs[n_states:])
            q_star_max = np.maximum(test_q[i][:n_states], test_q[i][n_states:])
            bias_max_task.append(np.mean(q_hat_max - q_star_max))

        js.append(j_task)
        diff_qs.append(diff_q_task)
        bias_sas.append(bias_sa_task)
        bias_maxs.append(bias_max_task)
        prev_dataset = dataset

    return js, diff_qs, bias_sas, bias_maxs, all_losses, all_pre_growth_losses, all_q_errors, all_metrics, fit_params


if __name__ == '__main__':
    args = tyro.cli(Args)
    assert args.growth_mode in GROWTH_MODULES, (
        f"Unknown growth_mode '{args.growth_mode}'. "
        f"Choose from: {list(GROWTH_MODULES)}"
    )

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(exp_id, args)
        for exp_id in range(args.n_exp)
    )

    Js = [o[0] for o in out]
    Qs = [o[1] for o in out]
    Bias_sa = [o[2] for o in out]
    Bias_max = [o[3] for o in out]
    Losses = [o[4] for o in out]
    Pre_growth_losses = [o[5] for o in out]
    Q_errors = [o[6] for o in out]
    all_exp_metrics = [o[7] for o in out]
    fit_params = out[0][8]

    boost = 'boosted' if args.use_boosting else 'no_boosted'
    cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
    subfolder = 'logs_curriculum' if args.use_curriculum else 'logs_no_curriculum'
    folder_name = (
        f'./logs/{subfolder}/grow_{args.growth_mode}_{boost}_{cur}'
        f'_h{args.initial_hidden}_ge{args.grow_every}'
        f'_lr{args.lr}_ep{args.n_epochs}_bs{args.batch_size}'
    )
    print(f"Output folder: {folder_name}")
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    config = asdict(args)
    config['ms'] = _ms_for_args(args)
    with open(folder_name + '/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    np.save(folder_name + '/J.npy', Js)
    np.save(folder_name + '/Q.npy', Qs)
    np.save(folder_name + '/bias_sa.npy', Bias_sa)
    np.save(folder_name + '/bias_max.npy', Bias_max)
    if args.monitor_loss:
        np.save(folder_name + '/losses.npy', np.array(Losses, dtype=float))
        np.save(
            folder_name + '/q_errors_per_epoch.npy',
            np.array(Q_errors, dtype=float),
        )
        if args.pre_growth_steps > 0 and Pre_growth_losses and Pre_growth_losses[0]:
            np.save(
                folder_name + '/pre_growth_losses.npy',
                np.array(Pre_growth_losses, dtype=float),
            )

    print('J: ', np.mean(Js, 0))
    print('Q diff: ', np.mean(Qs, 0))

    if all_exp_metrics and all_exp_metrics[0]:
        metric_keys = list(all_exp_metrics[0][0].keys())
        for key in metric_keys:
            vals = np.array(
                [[m.get(key, np.nan) for m in exp_metrics] for exp_metrics in all_exp_metrics],
                dtype=float,
            )
            np.save(folder_name + f'/metric_{key}.npy', vals)
