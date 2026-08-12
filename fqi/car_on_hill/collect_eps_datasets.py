"""
Collect datasets with epsilon-greedy policy (epsilon < 1) and analyze them.

Usage (from boosted_curriculum_rl/fqi/car_on_hill/):
    python collect_eps_datasets.py --epsilon 0.1
    python collect_eps_datasets.py --epsilon 0.3 --n-episodes 1000
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

    datasets_eps = {}
    for m in ms:
        print(f"\n{'='*50}\nm = {m:.3f}")
        mdp = CarOnHill()
        mdp._m = m

        # Load existing random dataset — needed for the 1 bootstrap fit
        with open('data/dataset_%1.3f.pkl' % m, 'rb') as f:
            dataset_random = pickle.load(f)
        print(f"  Loaded random dataset ({len(dataset_random)} transitions)")

        # One FQI fit to make ExtraTrees callable (bootstrap only, not training)
        pi = EpsGreedy(epsilon=Parameter(value=args.epsilon))
        agent = FQI(
            mdp.info, pi,
            FastExtraTreesActionRegressor,
            n_iterations=20,
            approximator_params=approximator_params,
            quiet=True,
        )
        agent.fit(dataset_random)

        # Collect with trained policy + epsilon
        core = Core(agent, mdp)
        print(f"  Collecting {args.n_episodes} episodes with epsilon={args.epsilon} ...")
        dataset_eps = core.evaluate(n_episodes=args.n_episodes, quiet=False)
        datasets_eps[m] = dataset_eps

        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        with open(f"{outdir}/dataset_{m:.3f}.pkl", 'wb') as f:
            pickle.dump(dataset_eps, f)
        print(f"  Saved {len(dataset_eps)} transitions -> {outdir}/dataset_{m:.3f}.pkl")

    run_analysis(
        datasets_eps,
        outdir=outdir,
        title=f"Car-on-Hill datasets (20-iter FQI + epsilon={args.epsilon})",
    )


if __name__ == '__main__':
    main()
