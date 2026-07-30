import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Roman"],
})


def plot_lines(ax, data, color):
    reshaped_data = np.concatenate([data[:, i, :] for i in range(0, data.shape[1])], axis=-1)
    mean = np.mean(reshaped_data, axis=0)
    std = np.std(reshaped_data, axis=0)
    n = mean.shape[0]
    x = np.linspace(1, n, n)

    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    l1, = ax.plot(x, mean, color=color, linewidth=1.5)

    return l1


def visualize_evolution(path=None, residuals=False, fontsize=6, ticksize=6, axsize=(0.17, 0.215, 0.825, 0.7),
                        yoffset=0.035, suffix=""):
    fig = plt.figure(figsize=(2.56, 1.5))
    ax = fig.add_axes(axsize)

    filename = "Q.npy" if residuals else "J.npy"
    suf = f"_{suffix}" if suffix else ""

    boosted_curriculum_data = np.load(os.path.join(f"logs/neural_boosted_curriculum{suf}", filename))
    boosted_data = np.load(os.path.join(f"logs/neural_boosted_no_curriculum{suf}", filename))
    curriculum_data = np.load(os.path.join(f"logs/neural_no_boosted_curriculum{suf}", filename))
    default_data = np.load(os.path.join(f"logs/neural_no_boosted_no_curriculum{suf}", filename))

    l1 = plot_lines(ax, boosted_curriculum_data, "C0")
    l2 = plot_lines(ax, boosted_data, "C1")
    l3 = plot_lines(ax, curriculum_data, "C2")
    l4 = plot_lines(ax, default_data, "C3")

    ylim = ax.get_ylim()
    plt.vlines([20, 40], *ylim, color="black", linestyle="--", linewidths=[2, 2], alpha=0.5)
    patch = Rectangle([0, ylim[0]], 20, ylim[1] - ylim[0], color="black", alpha=0.4)
    fig.gca().add_patch(patch)
    patch = Rectangle([20, ylim[0]], 20, ylim[1] - ylim[0], color="black", alpha=0.2)
    fig.gca().add_patch(patch)

    plt.text(10, ylim[1] + yoffset, r"$\mathcal{T}_1$", fontsize=ticksize)
    plt.text(30, ylim[1] + yoffset, r"$\mathcal{T}_2$", fontsize=ticksize)
    plt.text(50, ylim[1] + yoffset, r"$\mathcal{T}_3$", fontsize=ticksize)

    plt.legend([l1, l2, l3, l4], ["BC-NeuralFQI", "B-NeuralFQI", "C-NeuralFQI", "NeuralFQI"],
               fontsize=ticksize, ncol=2)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel(r"$\| Q_t^k - Q_t^* \|_{1, \mu}$" if residuals else "Cum. Disc. Return", fontsize=fontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)
    plt.xlim([0, 60])
    plt.ylim(ylim)

    plt.grid()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")


def get_suffixes():
    import glob
    folders = glob.glob("logs/neural_boosted_curriculum_*")
    suffixes = [os.path.basename(f).replace("neural_boosted_curriculum_", "") for f in folders]
    return suffixes if suffixes else [""]


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    for suffix in get_suffixes():
        visualize_evolution(path=f"figures/car_on_hill_neural_performance_{suffix}.pdf", residuals=False, fontsize=9,
                            axsize=(0.195, 0.215, 0.78, 0.7), yoffset=0.035, suffix=suffix)
        visualize_evolution(path=f"figures/car_on_hill_neural_diff_q_{suffix}.pdf", residuals=True, fontsize=9,
                            axsize=(0.17, 0.215, 0.805, 0.7), yoffset=0.02, suffix=suffix)


# import numpy as np
# import matplotlib.pyplot as plt

# J = np.load("logs/neural_no_boosted_curriculum/J.npy")
# Q = np.load("logs/neural_no_boosted_curriculum/Q.npy")

# J_flat = np.concatenate([J[:, i, :] for i in range(J.shape[1])], axis=-1)
# Q_flat = np.concatenate([Q[:, i, :] for i in range(Q.shape[1])], axis=-1)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# n = J_flat.shape[1]
# x = np.arange(1, n + 1)

# ax1.plot(x, J_flat[0], label="C-NeuralFQI")
# ax1.vlines([20, 40], *ax1.get_ylim(), colors="black", linestyles="--", alpha=0.5)
# ax1.set_xlabel("Iteration")
# ax1.set_ylabel("Cum. Disc. Return")
# ax1.set_title("J (neural_no_boosted_curriculum)")
# ax1.legend()
# ax1.grid()

# ax2.plot(x, Q_flat[0], label="C-NeuralFQI", color="C0")
# ax2.vlines([20, 40], *ax2.get_ylim(), colors="black", linestyles="--", alpha=0.5)
# ax2.set_xlabel("Iteration")
# ax2.set_ylabel("||Q - Q*||")
# ax2.set_title("Q diff (neural_no_boosted_curriculum)")
# ax2.legend()
# ax2.grid()

# plt.tight_layout()
# plt.savefig("figures/nc.pdf")
# plt.show()
