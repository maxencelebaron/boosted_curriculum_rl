import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Roman"],
})


def plot_loss_curves(axes, losses, n_tasks, use_curriculum, fontsize=9, ticksize=8):
    """
    axes: list of Axes — length n_tasks for curriculum, 1 otherwise.
    losses: (n_exp, n_iters, n_epochs)

    Curriculum: one subplot per task, each with its own y-axis scale.
    No curriculum: single subplot in Greens (T3). Boosting runs 3 stages on the
    same task (m=1.2), so a single continuous Greens gradient is used regardless.
    """
    n_iters = losses.shape[1]
    n_epochs = losses.shape[2]
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
                mean_curve = np.mean(losses[:, idx, :], axis=0)
                ax.plot(epochs, mean_curve, color=colors[i], alpha=0.8, linewidth=0.7)
            ax.set_title(fr"$\mathcal{{T}}_{task+1}$", fontsize=fontsize)
            ax.set_xlabel("Epoch", fontsize=fontsize)
            ax.set_ylabel("Loss", fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.grid()
    else:
        ax = axes[0]
        colors = plt.cm.Greens(np.linspace(0.35, 0.95, n_iters))
        for idx in range(n_iters):
            mean_curve = np.mean(losses[:, idx, :], axis=0)
            line, = ax.plot(epochs, mean_curve, color=colors[idx], alpha=0.8, linewidth=0.7)
        ax.legend([line], [r"$\mathcal{T}_3$"], fontsize=ticksize)
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel("Loss", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.grid()


def visualize_loss_convergence(folder, algo_label, n_tasks, path=None, fontsize=9, ticksize=8):
    losses = np.load(os.path.join(folder, "losses.npy"))
    use_curriculum = "no_curriculum" not in folder

    n_subplots = n_tasks if use_curriculum else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(3.5 * n_subplots, 3))
    if n_subplots == 1:
        axes = [axes]

    fig.suptitle(algo_label, fontsize=fontsize)
    plot_loss_curves(axes, losses, n_tasks, use_curriculum, fontsize=fontsize, ticksize=ticksize)

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