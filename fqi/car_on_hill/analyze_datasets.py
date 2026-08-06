import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

datasets = {}
for m in [0.800, 1.000, 1.200]:
    fname = 'data/dataset_%1.3f.pkl' % m
    with open(fname, 'rb') as f:
        datasets[m] = pickle.load(f)

def parse_dataset(dataset):
    states      = np.array([t[0] for t in dataset])
    actions     = np.array([t[1] for t in dataset]).flatten()
    rewards     = np.array([t[2] for t in dataset])
    absorbing   = np.array([t[4] for t in dataset])
    last        = np.array([t[5] for t in dataset])
    return states, actions, rewards, absorbing, last

lines = ["=" * 60]
for m, dataset in datasets.items():
    states, actions, rewards, absorbing, last = parse_dataset(dataset)
    n = len(dataset)
    n_episodes = last.sum()
    ep_lengths  = np.diff(np.where(last)[0], prepend=-1)

    pos_rew  = (rewards > 0).sum()
    neg_rew  = (rewards < 0).sum()
    zero_rew = (rewards == 0).sum()

    lines += [
        f"\nm = {m:.3f}",
        f"  Total transitions  : {n}",
        f"  Episodes           : {int(n_episodes)}",
        f"  Episode length     : mean={ep_lengths.mean():.1f}  min={ep_lengths.min()}  max={ep_lengths.max()}",
        f"  Rewards > 0 (+1)   : {pos_rew}  ({100*pos_rew/n:.1f}%)",
        f"  Rewards < 0 (-1)   : {neg_rew}  ({100*neg_rew/n:.1f}%)",
        f"  Rewards = 0        : {zero_rew} ({100*zero_rew/n:.1f}%)",
        f"  Absorbing states   : {absorbing.sum()}",
        f"  Actions 0/1        : {(actions==0).sum()} / {(actions==1).sum()}",
        f"  State[0] (pos) range : [{states[:,0].min():.3f}, {states[:,0].max():.3f}]",
        f"  State[1] (vel) range : [{states[:,1].min():.3f}, {states[:,1].max():.3f}]",
    ]
lines.append("=" * 60)

with open("data/dataset_analysis.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# Figures
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Car-on-Hill offline datasets (random policy)", fontsize=14)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors = {0.800: '#1f77b4', 1.000: '#ff7f0e', 1.200: '#2ca02c'}

for row, (m, dataset) in enumerate(datasets.items()):
    states, actions, rewards, absorbing, last = parse_dataset(dataset)
    c = colors[m]
    label = f"m={m:.1f}"

    # 1. Scatter plot of (position, velocity) colored by reward
    ax = fig.add_subplot(gs[row, 0])
    sc = ax.scatter(states[:, 0], states[:, 1], c=rewards, cmap='RdYlGn',
                    s=2, alpha=0.4, vmin=-1, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(f"{label} - state coverage")

    # 2. Reward distribution (bar)
    ax2 = fig.add_subplot(gs[row, 1])
    vals, cnts = np.unique(rewards, return_counts=True)
    ax2.bar([str(int(v)) for v in vals], cnts / len(rewards) * 100, color=c)
    ax2.set_xlabel("reward")
    ax2.set_ylabel("% transitions")
    ax2.set_title(f"{label} - reward distribution")

    # 3. Episode-length histogram
    ep_lengths = np.diff(np.where(last)[0], prepend=-1)
    ax3 = fig.add_subplot(gs[row, 2])
    ax3.hist(ep_lengths, bins=30, color=c, edgecolor='white', linewidth=0.4)
    ax3.set_xlabel("episode length (steps)")
    ax3.set_ylabel("count")
    ax3.set_title(f"{label} - episode lengths")


plt.savefig("data/dataset_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
