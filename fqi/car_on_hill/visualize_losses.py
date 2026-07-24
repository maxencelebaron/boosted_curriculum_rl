import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Roman"],
})


def plot_loss_curves(ax, losses, n_tasks, fontsize=9, ticksize=8):
    """
    losses: (n_exp, n_iters, n_epochs)
    One curve per outer iteration (mean over experiments), colored by task.
    """
    n_iters = losses.shape[1]
    n_epochs = losses.shape[2]
    iters_per_task = n_iters // n_tasks
    epochs = np.arange(1, n_epochs + 1)

    task_cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]
    task_handles = []

    for task in range(n_tasks):
        start = task * iters_per_task
        end = (task + 1) * iters_per_task
        colors = task_cmaps[task % len(task_cmaps)](np.linspace(0.35, 0.95, iters_per_task))
        for i, idx in enumerate(range(start, end)):
            mean_curve = np.mean(losses[:, idx, :], axis=0)
            line, = ax.plot(epochs, mean_curve, color=colors[i], alpha=0.8, linewidth=0.7)
        task_handles.append(line)

    ax.legend(task_handles, [fr"$\mathcal{{T}}_{t+1}$" for t in range(n_tasks)],
              fontsize=ticksize)
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid()


def visualize_loss_convergence(folder, algo_label, n_tasks, path=None, fontsize=9, ticksize=8):
    losses = np.load(os.path.join(folder, "losses.npy"))

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes((0.15, 0.15, 0.78, 0.72))
    ax.set_title(algo_label, fontsize=fontsize)

    plot_loss_curves(ax, losses, n_tasks, fontsize=fontsize, ticksize=ticksize)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    algorithms = [
        ("logs/neural_boosted_curriculum",       "BC-NeuralFQI", 3),
        ("logs/neural_boosted_no_curriculum",    "B-NeuralFQI",  3),
        ("logs/neural_no_boosted_curriculum",    "C-NeuralFQI",  3),
        ("logs/neural_no_boosted_no_curriculum", "NeuralFQI",    1),
    ]
    for folder, label, n_tasks in algorithms:
        visualize_loss_convergence(
            folder, label, n_tasks,
            path=f"figures/losses_{label}.pdf",
            fontsize=9, ticksize=8,
        )
