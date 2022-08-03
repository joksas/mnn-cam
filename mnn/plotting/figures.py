import matplotlib.pyplot as plt

from mnn import simulations
from mnn.expdata import load

from . import utils

AXIS_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 8
TICKS_FONT_SIZE = 8
SUBPLOT_LABEL_SIZE = 12
LINEWIDTH = 0.75
ONE_COLUMN_WIDTH = 8.5 / 2.54
TWO_COLUMNS_WIDTH = 17.8 / 2.54


def discretisation_boxplots():
    fig, axes = plt.subplots(figsize=(TWO_COLUMNS_WIDTH, 0.4 * TWO_COLUMNS_WIDTH))
    fig.tight_layout()

    configs = simulations.discretisation.configs()

    colors = [utils.color_dict()[key] for key in ["blue", "vermilion", "orange"]]

    boxplots = []
    for idx, config in enumerate(configs):
        accuracy = 100 * config.accuracy()
        error = 100 - accuracy
        color = colors[idx % 3]
        bplot = plt.boxplot(error, positions=[idx // 3 + idx], widths=[0.5], sym=color)
        boxplots.append(bplot)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.75)

    axes.set_yscale("log")
    plt.xticks([1, 5, 9], ["MNIST", "Fashion MNIST", "KMNIST"])
    plt.xlabel("Dataset", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Error (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    utils.add_boxplot_legend(
        axes,
        boxplots,
        ["Digital", "Memristive (32 states)", "Memristive (370 states)"],
        loc="upper left",
    )

    utils.save_fig(fig, "discretisation-boxplots")


def plot_discrete_levels(filename):
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.75 * ONE_COLUMN_WIDTH))
    fig.tight_layout()

    levels = load.retention_conductance_levels(filename)
    plt.hlines(levels, 0, 1, linewidth=0.25)

    plt.ylabel("Mean conductance (S)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="y", which="both", labelsize=TICKS_FONT_SIZE)
    axes.set_xticklabels([])
    axes.set_xticks([])

    utils.save_fig(fig, "discrete-levels")
