import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

os.environ["MPLCONFIGDIR"] = "/hpc_share/ee217092/cache"


def main():
    with open("stats_rerun.pkl", "rb") as f:
        stats = pickle.load(f)

    if isinstance(stats, tuple) and isinstance(stats[0], dict):
        _, _, stats = stats
    else:
        print("Expected a dictionary but got:", type(stats))
        return

    adversarial_losses = stats.get("target_losses")
    train_losses = stats.get("train_losses")
    step_size = stats.get("learning_rates")

    epochs = range(0, len(adversarial_losses))

    fig, ax1 = plt.subplots(figsize=(6, 4.5))

    # Plot Step-size on left Y-axis
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Step-size")
    ax1.set_yscale("log")
    ax1.plot(epochs, step_size, color=color, label="Step-size")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=10))
    ax1.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Plot adversarial loss on right Y-axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    color2 = "tab:green"
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.plot(epochs, adversarial_losses, color=color, label="Adversarial Loss")
    ax2.plot(epochs, train_losses, color=color2, label="Training Loss")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=10))
    ax2.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legends and save the plot
    fig.tight_layout()
    fig.legend(
        loc="upper center",  # 中央上部
        bbox_to_anchor=(0.5, 0),  # 図の外の下に配置
        ncol=3,  # 凡例を横に並べる
    )
    fig.tight_layout()
    plt.savefig("step-sizes_losses.png", dpi=300, bbox_inches="tight")
    print(epochs)


if __name__ == "__main__":
    main()
