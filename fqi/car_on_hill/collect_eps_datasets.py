"""
Collect datasets with epsilon-greedy policy (epsilon < 1) and analyze them.

Usage (from boosted_curriculum_rl/fqi/car_on_hill/):
    python collect_eps_datasets.py --epsilon 0.1
    python collect_eps_datasets.py --epsilon 0.3
    --n-episodes 1000
"""

import argparse
import pathlib
import pickle
import numpy as np

from fqi.fast_extra_tress import FastExtraTreesActionRegressor
from fqi.fqi import FQI
from mushroom_rl.core import Core
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

from analyze_datasets import run_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon",    type=float, default=0.1)
    parser.add_argument("--n-episodes", type=int,   default=1000)
    parser.add_argument("--n-fqi-iters", type=int,  default=20)
    parser.add_argument("--seed",       type=int,   default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    ms = [0.800, 1.000, 1.200]

    approximator_params = dict(
        input_shape=(2,),
        n_actions=2,
        n_models=1,
        n_estimators=50,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=args.seed,
        prediction='sum',
    )

    eps_tag = f"eps{args.epsilon:.2f}".replace('.', 'p')
    outdir = f"data_{eps_tag}"

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    datasets_collected = {}
    prev_agent = None

    for i, m in enumerate(ms):
        print(f"\n{'='*50}\nTask {i}  m = {m:.3f}")
        mdp = CarOnHill()
        mdp._m = m

        if i == 0:
            # No prior policy - collect with random policy
            pi_rand = EpsGreedy(epsilon=Parameter(value=args.epsilon))
            rand_agent = FQI(
                mdp.info, pi_rand,
                FastExtraTreesActionRegressor,
                n_iterations=1,
                approximator_params=approximator_params,
                quiet=True,
            )
            print(f" Collecting {args.n_episodes} episodes with random policy ...")
            dataset = Core(rand_agent, mdp).evaluate(n_episodes=args.n_episodes)
        else:
            # Collect with the policy learned on the previous task
            print(f"  Collecting {args.n_episodes} episodes with epsilon={args.epsilon} ...")
            dataset = Core(prev_agent, mdp).evaluate(n_episodes=args.n_episodes)

        with open(f"{outdir}/dataset_{m:.3f}.pkl", 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  Saved {len(dataset)} transitions -> {outdir}/dataset_{m:.3f}.pkl")
        datasets_collected[m] = dataset

        if i < len(ms) - 1:
            # Train FQI - policy for the next task (inutile pour la dernière tâche)
            pi = EpsGreedy(epsilon=Parameter(value=args.epsilon))
            prev_agent = FQI(
                mdp.info,
                pi,
                FastExtraTreesActionRegressor,
                n_iterations=1,
                approximator_params=approximator_params,
                quiet=True,
            )
            print(f"  Training FQI ({args.n_fqi_iters} iters) on m={m:.3f} ...")
            for _ in range(args.n_fqi_iters):
                prev_agent.fit(dataset)

    run_analysis(
        datasets_collected,
        outdir=outdir,
        title=f"Car-on-Hill datasets (curriculum collection, epsilon={args.epsilon})",
    )


if __name__ == '__main__':
    main()
