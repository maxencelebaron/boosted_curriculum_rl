import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
})


def _safe_load(path):
    try:
        return np.load(path)
    except FileNotFoundError:
        return None


def _load_per_seed(log_dir, filename):
    """Stack J-0.npy, J-1.npy, ... into (n_seeds, n_eval_points)."""
    stem = filename.replace('.npy', '')
    seeds, i = [], 0
    while True:
        arr = _safe_load(os.path.join(log_dir, f'{stem}-{i}.npy'))
        if arr is None:
            break
        seeds.append(arr)
        i += 1
    return np.stack(seeds, axis=0) if seeds else None


def plot_lines(ax, data, color):
    """data: (n_seeds, n_tasks, n_eval_points) or (n_seeds, n_eval_points)"""
    if data.ndim == 2:
        data = data[:, np.newaxis, :]
    reshaped = np.concatenate([data[:, i, :] for i in range(data.shape[1])], axis=-1)
    mean = np.mean(reshaped, axis=0)
    std = np.std(reshaped, axis=0)
    n = mean.shape[0]
    x = np.linspace(1, n, n)

    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    l, = ax.plot(x, mean, color=color, linewidth=1.5)
    return l, mean


def _add_task_patches(ax, ylim, n_eval_per_task, n_tasks, yoffset, ticksize):
    boundaries = [n_eval_per_task * k for k in range(1, n_tasks)]
    if boundaries:
        ax.vlines(boundaries, *ylim, color="black", linestyle="--",
                  linewidths=[2] * len(boundaries), alpha=0.5)
    alphas = [0.4, 0.2, 0.0]
    for k in range(n_tasks - 1):
        patch = Rectangle(
            [n_eval_per_task * k, ylim[0]],
            n_eval_per_task, ylim[1] - ylim[0],
            color="black", alpha=alphas[k],
        )
        ax.add_patch(patch)
    task_labels = [r"$\mathcal{T}_1$", r"$\mathcal{T}_2$", r"$\mathcal{T}_3$"]
    for k in range(n_tasks):
        ax.text(
            n_eval_per_task * k + n_eval_per_task / 2,
            ylim[1] + yoffset,
            task_labels[k], fontsize=ticksize, ha='center',
        )


def _plot_metric(filename, ylabel, path, fontsize, ticksize, axsize, yoffset,
                 n_eval_per_task, n_tasks, ylim=None):
    log_dir = "logs/dqn_car_on_hill"
    data = _safe_load(os.path.join(log_dir, filename)) or _load_per_seed(log_dir, filename)
    datasets = [(data, "C0", "DQN")]

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes(axsize)

    lines, means, labels = [], [], []
    for data, color, label in datasets:
        if data is not None:
            l, m = plot_lines(ax, data, color)
            lines.append(l)
            means.append(m)
            labels.append(label)

    if not means:
        print(f"No data found for {filename}, skipping.")
        plt.close()
        return

    if ylim is None:
        all_means = np.concatenate(means)
        lo, hi = np.percentile(all_means, 2), np.percentile(all_means, 98)
        margin = 0.05 * (hi - lo)
        ylim = (lo - margin, hi + margin)

    _add_task_patches(ax, ylim, n_eval_per_task, n_tasks, yoffset, ticksize)

    ax.legend(lines, labels, fontsize=ticksize, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.18), framealpha=0.75)
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.set_xlim([0, n_eval_per_task * n_tasks])
    ax.set_ylim(ylim)
    ax.grid()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")
        print("Saved to", path)
    plt.close()


def visualize_evolution(path=None, residuals=False, fontsize=6, ticksize=6,
                        axsize=(0.17, 0.215, 0.825, 0.7), yoffset=0.035,
                        n_eval_per_task=60, n_tasks=1, ylim=None):
    filename = "Q.npy" if residuals else "J.npy"
    ylabel = r"$\| Q - Q^* \|_{1}$" if residuals else "Cum. Disc. Return"
    _plot_metric(filename, ylabel, path, fontsize, ticksize, axsize, yoffset,
                 n_eval_per_task, n_tasks, ylim)


def visualize_overestimation(path=None, metric="sa", fontsize=6, ticksize=6,
                             axsize=(0.22, 0.215, 0.755, 0.7), yoffset=0.035,
                             n_eval_per_task=60, n_tasks=1, ylim=None):
    filename = "bias_sa.npy" if metric == "sa" else "bias_max.npy"
    if metric == "sa":
        ylabel = r"$\frac{1}{N}\sum_{s,a}(\hat{Q}(s,a) - Q^*(s,a))$"
    else:
        ylabel = r"$\frac{1}{|S|}\sum_s(\max_a \hat{Q} - \max_a Q^*)$"
    _plot_metric(filename, ylabel, path, fontsize, ticksize, axsize, yoffset,
                 n_eval_per_task, n_tasks, ylim)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    visualize_evolution(
        path="figures/car_on_hill_dqn_performance.pdf",
        residuals=False, fontsize=9,
        axsize=(0.195, 0.215, 0.78, 0.7), yoffset=0.035,
    )
    visualize_evolution(
        path="figures/car_on_hill_dqn_diff_q.pdf",
        residuals=True, fontsize=9,
        axsize=(0.17, 0.215, 0.805, 0.7), yoffset=0.02,
    )
    visualize_overestimation(
        path="figures/car_on_hill_dqn_bias_sa.pdf",
        metric="sa", fontsize=9,
        axsize=(0.22, 0.215, 0.755, 0.7), yoffset=0.02,
    )
    visualize_overestimation(
        path="figures/car_on_hill_dqn_bias_max.pdf",
        metric="max", fontsize=9,
        axsize=(0.22, 0.215, 0.755, 0.7), yoffset=0.02,
    )