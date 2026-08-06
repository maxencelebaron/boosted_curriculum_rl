import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Roman"],
})

BASELINE_COLOR = "C3"
GROW_COLORS = ["C0", "C1", "C2", "C4", "C5"]


def plot_lines(ax, data, color, label):
    reshaped = np.concatenate(
        [data[:, i, :] for i in range(data.shape[1])], axis=-1
    )
    mean = np.mean(reshaped, axis=0)
    std = np.std(reshaped, axis=0)
    x = np.linspace(1, mean.shape[0], mean.shape[0])
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    l, = ax.plot(x, mean, color=color, linewidth=1.5, label=label)
    return l, mean


def extract_grow_label(folder):
    name = os.path.basename(folder)
    return name.replace("grow_", "").split("_no_boosted")[0]


def extract_neural_label(folder):
    name = os.path.basename(folder)
    rest = name[len("neural_"):]  # "no_boosted_curriculum_..." or "boosted_no_curriculum_..."
    boosted = rest.startswith("boosted")
    after = rest[len("no_boosted_"):] if rest.startswith("no_boosted") else rest[len("boosted_"):]
    curriculum = after.startswith("curriculum")
    if boosted and curriculum:
        return "BC-NeuralFQI"
    if boosted:
        return "B-NeuralFQI"
    if curriculum:
        return "C-NeuralFQI"
    return "NeuralFQI"


def plot_grow_results(
    logs_subdir,
    curriculum,
    filename,
    ylabel,
    path=None,
    fontsize=9,
    ticksize=6,
    axsize=(0.17, 0.215, 0.825, 0.7),
    yoffset=0.035,
    ylim=None,
    legend_loc="best",
):
    all_folders = sorted(glob.glob(os.path.join(logs_subdir, "*")))
    all_folders = [f for f in all_folders if os.path.isdir(f)]

    if not all_folders:
        print(f"No folders found in {logs_subdir}")
        return

    fig = plt.figure(figsize=(2.56, 1.5))
    ax = fig.add_axes(axsize)

    all_means = []
    growth_markers = []  # list of (growth_iters, color)
    grow_idx = 0
    for folder in all_folders:
        npy_path = os.path.join(folder, filename)
        if not os.path.exists(npy_path):
            continue
        name = os.path.basename(folder)
        data = np.load(npy_path)
        if name.startswith("neural_"):
            _, m = plot_lines(ax, data, BASELINE_COLOR, extract_neural_label(folder))
            all_means.append(m)
        elif name.startswith("grow_"):
            color = GROW_COLORS[grow_idx % len(GROW_COLORS)]
            label = extract_grow_label(folder)
            _, m = plot_lines(ax, data, color, label)
            all_means.append(m)

            grew_path = os.path.join(folder, "metric_grew.npy")
            if os.path.exists(grew_path):
                grew = np.load(grew_path).astype(float)  # (n_exp, n_iters)
                # iterations (1-indexed) where at least one experiment grew
                grew_iters = np.where(np.nanmean(grew, axis=0) > 0)[0] + 1
                growth_markers.append((grew_iters, color))

            grow_idx += 1

    if ylim is None:
        if all_means:
            all_m = np.concatenate(all_means)
            lo, hi = np.percentile(all_m, 2), np.percentile(all_m, 98)
            margin = 0.05 * (hi - lo)
            ylim = (lo - margin, hi + margin)
        else:
            ylim = ax.get_ylim()

    all_growth_iters = sorted({it for grew_iters, _ in growth_markers for it in grew_iters})
    if all_growth_iters:
        ax.vlines(all_growth_iters, *ylim, color="indigo", linestyle="--", linewidth=0.7, alpha=0.6)

    if curriculum:
        plt.vlines(
            [20, 40], *ylim,
            color="black", linestyle="--", linewidths=[2, 2], alpha=0.5,
        )
        ax.add_patch(Rectangle(
            [0, ylim[0]], 20, ylim[1] - ylim[0], color="black", alpha=0.4,
        ))
        ax.add_patch(Rectangle(
            [20, ylim[0]], 20, ylim[1] - ylim[0], color="black", alpha=0.2,
        ))
        plt.text(10, ylim[1] + yoffset, r"$\mathcal{T}_1$", fontsize=ticksize)
        plt.text(30, ylim[1] + yoffset, r"$\mathcal{T}_2$", fontsize=ticksize)
        plt.text(50, ylim[1] + yoffset, r"$\mathcal{T}_3$", fontsize=ticksize)

    plt.legend(fontsize=ticksize, ncol=2, loc=legend_loc, framealpha=0.75)
    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.gca().tick_params(axis="both", which="major", labelsize=ticksize)
    plt.xlim([0, 60])
    plt.ylim(ylim)
    plt.grid()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    for curriculum, subfolder in [
        (True,  "logs/logs_curriculum"),
        (False, "logs/logs_no_curriculum"),
    ]:
        tag = "curriculum" if curriculum else "no_curriculum"

        plot_grow_results(
            subfolder,
            curriculum,
            filename="J.npy",
            ylabel="Cum. Disc. Return",
            path=f"figures/car_on_hill_grow_performance_{tag}.pdf",
            fontsize=9, ticksize=6,
            axsize=(0.195, 0.215, 0.78, 0.7), yoffset=0.035,
        )
        plot_grow_results(
            subfolder,
            curriculum,
            filename="Q.npy",
            ylabel=r"$\| Q_t^k - Q_t^* \|_{1, \mu}$",
            path=f"figures/car_on_hill_grow_diff_q_{tag}.pdf",
            fontsize=9, ticksize=6,
            axsize=(0.17, 0.215, 0.805, 0.7), yoffset=0.02,
        )
        plot_grow_results(
            subfolder,
            curriculum,
            filename="bias_sa.npy",
            ylabel=r"$\frac{1}{N}\sum_{s,a}(\hat{Q}(s,a) - Q^*(s,a))$",
            path=f"figures/car_on_hill_grow_bias_sa_{tag}.pdf",
            fontsize=9, ticksize=6,
            axsize=(0.22, 0.215, 0.755, 0.7), yoffset=0.02,
        )
        plot_grow_results(
            subfolder,
            curriculum,
            filename="bias_max.npy",
            ylabel=r"$\frac{1}{|S|}\sum_s(\max_a \hat{Q}(s,a) - \max_a Q^*(s,a))$",
            path=f"figures/car_on_hill_grow_bias_max_{tag}.pdf",
            fontsize=9, ticksize=6,
            axsize=(0.22, 0.215, 0.755, 0.7), yoffset=0.02,
        )
