import os
from pathlib import Path

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
    config = simulations.discretisation.get_config()

    accuracy = 100 * config.test_accuracy()
    boxplot_accuracies = [accuracy[:, 0, idx] for idx in [0, 1]]
    boxplot_colors = [utils.color_dict()[key] for key in ["blue", "orange"]]

    for idx, (boxplot_accuracy, boxplot_color) in enumerate(
        zip(boxplot_accuracies, boxplot_colors)
    ):
        bplot = plt.boxplot(boxplot_accuracy, positions=[idx], sym=boxplot_color)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=boxplot_color, linewidth=0.5)

    plt.xticks([0, 1], ["Digital", "Memristive"])
    plt.xlabel("Weight implementation", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", which="both", labelsize=TICKS_FONT_SIZE)

    path = os.path.join(Path(__file__).parent, "discretisation-boxplots.pdf")
    plt.savefig(path, bbox_inches="tight", transparent=True)


def plot_discrete_levels():
    fig, axes = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.75 * ONE_COLUMN_WIDTH))
    fig.tight_layout()

    levels = load.retention_conductance_levels()
    plt.hlines(levels, 0, 1, linewidth=0.25)

    plt.ylabel("Mean conductance (S)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="y", which="both", labelsize=TICKS_FONT_SIZE)
    axes.set_xticklabels([])
    axes.set_xticks([])

    path = os.path.join(Path(__file__).parent, "discrete-levels.pdf")
    plt.savefig(path, bbox_inches="tight", transparent=True)
