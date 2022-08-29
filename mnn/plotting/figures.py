import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnn import simulations
from mnn.expdata import load

from . import utils

plt.style.use(os.path.join(Path(__file__).parent.absolute(), "style.mplstyle"))

ONE_COLUMN_WIDTH = 8.5 / 2.54
TWO_COLUMNS_WIDTH = 17.8 / 2.54


def discretisation_boxplots():
    fig, axes = plt.subplots(figsize=(0.66 * TWO_COLUMNS_WIDTH, 0.4 * TWO_COLUMNS_WIDTH))
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
    plt.xticks([1, 5], ["MNIST", "Fashion MNIST"])
    plt.xlabel("Dataset")
    plt.ylabel("Error (%)")
    plt.tick_params(axis="both", which="both")

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

    plt.ylabel("Mean conductance (S)")
    plt.tick_params(axis="y", which="both")
    axes.set_xticklabels([])
    axes.set_xticks([])

    utils.save_fig(fig, "discrete-levels")

def plot_sweeps(filename):
    fig, axes = plt.subplots(figsize=(TWO_COLUMNS_WIDTH, 0.4 * TWO_COLUMNS_WIDTH))
    fig.tight_layout()

    sweeps = load.sweeping_data(filename)
    for sweep in sweeps:
        plt.plot(sweep[:, 0], sweep[:, 1], linestyle="dashed")

    axes.set_xlabel("Voltage (V)")
    axes.set_ylabel("Current (A)")

    axes.set_ylim(bottom=0)

    utils.save_fig(fig, "sweeps")


def training():
    fig, axes = plt.subplots(
        1, 3, figsize=(TWO_COLUMNS_WIDTH, 0.4 * TWO_COLUMNS_WIDTH), sharex=True, sharey=True
    )
    fig.tight_layout()

    configs = [
        simulations.training.mnist(),
        simulations.training.fashion_mnist(),
    ]
    datasets = ["MNIST", "Fashion MNIST", "KMNIST"]
    epochs = np.arange(1, 1001)

    colors = utils.color_dict()
    for axis, config, dataset in zip(axes, configs, datasets):
        for _ in range(config.num_repeats):
            training_error = 100 * (
                1 - np.array(config.info()["training_data"]["history"]["accuracy"])
            )
            validation_error = 100 * (
                1 - np.array(config.info()["training_data"]["history"]["val_accuracy"])
            )
            axis.plot(
                epochs,
                training_error,
                label="Training",
                color=colors["orange"],
            )
            axis.plot(
                epochs,
                validation_error,
                label="Validation",
                color=colors["blue"],
            )
            config.next_iteration()
        axis.set_title(dataset)
        axis.set_xlabel("Epoch")

    axes[0].set_ylabel("Error (%)")
    axes[0].set_yscale("log")
    axes[0].set_xlim([0, 1000])

    utils.save_fig(fig, "training")
