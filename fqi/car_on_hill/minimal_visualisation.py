import numpy as np
import matplotlib.pyplot as plt

J = np.load("logs/neural_boosted_curriculum/J.npy")
Q = np.load("logs/neural_boosted_curriculum/Q.npy")

J_flat = np.concatenate([J[:, i, :] for i in range(J.shape[1])], axis=-1)
Q_flat = np.concatenate([Q[:, i, :] for i in range(Q.shape[1])], axis=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

n = J_flat.shape[1]
x = np.arange(1, n + 1)

ax1.plot(x, J_flat[0], label="BC-NeuralFQI")
ax1.vlines([20, 40], *ax1.get_ylim(), colors="black", linestyles="--", alpha=0.5)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cum. Disc. Return")
ax1.set_title("J (neural_boosted_curriculum)")
ax1.legend()
ax1.grid()

ax2.plot(x, Q_flat[0], label="BC-NeuralFQI", color="C0")
ax2.vlines([20, 40], *ax2.get_ylim(), colors="black", linestyles="--", alpha=0.5)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("||Q - Q*||")
ax2.set_title("Q diff (neural_boosted_curriculum)")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.savefig("figures/partial_results.pdf")
plt.show()
