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
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.75 * ONE_COLUMN_WIDTH))
    fig.tight_layout()

    configs = [
        simulations.discretisation.ideal_config(),
        simulations.discretisation.nonideal_config(),
    ]

    colors = [utils.color_dict()[key] for key in ["blue", "orange"]]

    for idx, (config, color) in enumerate(zip(configs, colors)):
        accuracy = 100 * config.accuracy()
        bplot = plt.boxplot(accuracy, positions=[idx], sym=color)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.5)

    plt.xticks([0, 1], ["Digital", "Memristive"])
    plt.xlabel("Weight implementation", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

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
