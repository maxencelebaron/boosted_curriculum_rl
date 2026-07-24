import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Roman"],
})


def plot_curves(
    axes,
    data,
    n_tasks,
    use_curriculum,
    ylabel,
    show_task_title=False,
    fontsize=9,
    ticksize=8
):
    """
    axes: list of Axes — length n_tasks for curriculum, 1 otherwise.
    data: (n_exp, n_iters, n_epochs)

    Curriculum: one subplot per task, each with its own y-axis scale.
    No curriculum: single subplot in Greens (T3).
    """
    n_iters = data.shape[1]
    n_epochs = data.shape[2]
    epochs = np.arange(1, n_epochs + 1)
    task_cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

    if use_curriculum:
        iters_per_task = n_iters // n_tasks
        for task in range(n_tasks):
            ax = axes[task]
            start = task * iters_per_task
            end = (task + 1) * iters_per_task
            colors = task_cmaps[task % len(task_cmaps)](np.linspace(0.35, 0.95, iters_per_task))
            for i, idx in enumerate(range(start, end)):
                mean_curve = np.mean(data[:, idx, :], axis=0)
                ax.plot(epochs, mean_curve, color=colors[i], alpha=0.8, linewidth=0.7)
            if show_task_title:
                ax.set_title(fr"$\mathcal{{T}}_{task+1}$", fontsize=fontsize)
            ax.set_xlabel("Epoch", fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.grid()
    else:
        ax = axes[0]
        colors = plt.cm.Greens(np.linspace(0.35, 0.95, n_iters))
        for idx in range(n_iters):
            mean_curve = np.mean(data[:, idx, :], axis=0)
            line, = ax.plot(epochs, mean_curve, color=colors[idx], alpha=0.8, linewidth=0.7)
        ax.legend([line], [r"$\mathcal{T}_3$"], fontsize=ticksize)
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.grid()


def visualize_loss_convergence(folder, algo_label, n_tasks, path=None, fontsize=9, ticksize=8):
    losses = np.load(os.path.join(folder, "losses.npy"))
    q_errors = np.load(os.path.join(folder, "q_errors_per_epoch.npy"))
    use_curriculum = "no_curriculum" not in folder

    n_subplots = n_tasks if use_curriculum else 1
    fig, axes = plt.subplots(2, n_subplots, figsize=(3.5 * n_subplots, 5.5), squeeze=False)

    fig.suptitle(algo_label, fontsize=fontsize)
    plot_curves(axes[0], losses, n_tasks, use_curriculum,
                ylabel="Loss", show_task_title=True, fontsize=fontsize, ticksize=ticksize)
    plot_curves(axes[1], q_errors, n_tasks, use_curriculum,
                ylabel=r"$\| Q_t^{k_j} - Q_t^* \|_{1, \mu}$", show_task_title=False, fontsize=fontsize, ticksize=ticksize)

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    algorithms = [
        ("logs/neural_boosted_curriculum",       "BC-NeuralFQI", 3),
        ("logs/neural_boosted_no_curriculum",    "B-NeuralFQI",  1),
        ("logs/neural_no_boosted_curriculum",    "C-NeuralFQI",  3),
        ("logs/neural_no_boosted_no_curriculum", "NeuralFQI",    1),
    ]
    for folder, label, n_tasks in algorithms:
        visualize_loss_convergence(
            folder, label, n_tasks,
            path=f"figures/losses_{label}.pdf",
            fontsize=9, ticksize=8,
        )
