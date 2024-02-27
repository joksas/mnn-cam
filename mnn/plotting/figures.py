import os
from pathlib import Path

import matplotlib
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
    num_sizes = len(configs) // 2

    cmap = matplotlib.cm.get_cmap("plasma")
    # Discrete color
    colors = [utils.color_dict()["blue"]]
    # Memristive colors
    fractions = np.linspace(0.7, 0.1, num=num_sizes - 1)
    for fraction in fractions:
        colors.append(matplotlib.colors.rgb2hex(cmap(fraction)))

    boxplots = []
    for idx, config in enumerate(configs):
        accuracy = 100 * config.accuracy()
        error = 100 - accuracy
        color = colors[idx % num_sizes]
        bplot = plt.boxplot(error, positions=[idx // num_sizes + idx], widths=[0.5], sym=color)
        boxplots.append(bplot)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.75)

    axes.set_yscale("log")
    plt.xticks([2.5, 9.5], ["MNIST", "Fashion MNIST"])
    plt.xlabel("Dataset")
    plt.ylabel("Error (%)")
    plt.tick_params(axis="both", which="both")

    utils.add_boxplot_legend(
        axes,
        boxplots,
        [
            "Digital",
            "Memristive (32 states)",
            "Memristive (128 states)",
            "Memristive (303 states)",
            "Memristive (370 states)",
            "Memristive (526 states)",
        ],
        loc="upper left",
    )

    utils.save_fig(fig, "discretisation-boxplots")


def discretisation_and_d2d_boxplots():
    fig, axes = plt.subplots(figsize=(0.66 * TWO_COLUMNS_WIDTH, 0.4 * TWO_COLUMNS_WIDTH))
    fig.tight_layout()

    configs = simulations.discretisation_and_d2d.configs()
    num_sizes = len(configs) // 2

    cmap = matplotlib.cm.get_cmap("plasma")
    # Discrete color
    colors = [utils.color_dict()["blue"]]
    # Memristive colors
    fractions = np.linspace(0.7, 0.1, num=num_sizes - 1)
    for fraction in fractions:
        colors.append(matplotlib.colors.rgb2hex(cmap(fraction)))

    boxplots = []
    for idx, config in enumerate(configs):
        accuracy = 100 * config.accuracy()
        error = 100 - accuracy
        color = colors[idx % num_sizes]
        error = error.flatten()
        bplot = plt.boxplot(error, positions=[idx // num_sizes + idx], widths=[0.5], sym=color)
        boxplots.append(bplot)
        plt.setp(bplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bplot[element], color=color, linewidth=0.75)

    axes.set_yscale("log")
    plt.xticks([2.5, 9.5], ["MNIST", "Fashion MNIST"])
    plt.xlabel("Dataset")
    plt.ylabel("Error (%)")
    plt.tick_params(axis="both", which="both")

    utils.add_boxplot_legend(
        axes,
        boxplots,
        [
            "Digital",
            "Memristive (32 states)",
            "Memristive (128 states)",
            "Memristive (303 states)",
            "Memristive (370 states)",
            "Memristive (526 states)",
        ],
        loc="upper left",
    )

    utils.save_fig(fig, "discretisation-and-d2d-boxplots")


def lognormal_fit(low_R_data, high_R_data, mu_fit_params, sigma_fit_params):
        low_R_resistances = [low_R_point[0] for low_R_point in low_R_data]
        low_R_mus = [low_R_point[1] for low_R_point in low_R_data]
        low_R_sigmas = [low_R_point[2] for low_R_point in low_R_data]

        high_R_resistances = [high_R_point[0] for high_R_point in high_R_data]
        high_R_sigmas = [high_R_point[2] for high_R_point in high_R_data]
        high_R_mus = [high_R_point[1] for high_R_point in high_R_data]

        fig, axes = plt.subplots(
            2, 1, figsize=(TWO_COLUMNS_WIDTH, 0.75 * TWO_COLUMNS_WIDTH), sharex=True, sharey=False
        )

        axes[0].scatter(low_R_resistances, low_R_mus, color=utils.color_dict()["blue"])
        axes[0].scatter(high_R_resistances, high_R_mus, color=utils.color_dict()["orange"])

        axes[1].scatter(low_R_resistances, low_R_sigmas, color=utils.color_dict()["blue"])
        axes[1].scatter(high_R_resistances, high_R_sigmas, color=utils.color_dict()["orange"])

        full_range_x = np.linspace(1e3, 1e8, 1000)

        axes[0].plot(full_range_x, mu_fit_params[0] * np.log(full_range_x) + mu_fit_params[1], color=utils.color_dict()["black"], linestyle="--")
        axes[1].plot(full_range_x, sigma_fit_params[0] * np.log(full_range_x) + sigma_fit_params[1], color=utils.color_dict()["black"], linestyle="--")

        axes[0].set_xscale("log")
        axes[0].set_xlim([1e3, 1e8])

        axes[0].set_ylim(bottom=0)
        axes[1].set_ylim(bottom=0)

        axes[0].set_ylabel("Lognormal $\mu$ parameter")
        axes[1].set_ylabel("Lognormal $\sigma$ parameter")

        axes[1].set_xlabel("Average C2C resistance ($\Omega$)")

        axes[0].legend(
                ["LRS", "HRS"],
                loc="center",
                bbox_to_anchor=(0.5, 1.05),
                frameon=False,
                ncol=2,
                )

        fig.tight_layout()

        utils.save_fig(fig, "lognormal-fit")



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


def training():
    NUM_EPOCHS = 2
    fig, axes = plt.subplots(
        1, 3, figsize=(TWO_COLUMNS_WIDTH, 0.4 * TWO_COLUMNS_WIDTH), sharex=True, sharey=True
    )
    fig.tight_layout()

    configs = [
        simulations.training.mnist(),
        simulations.training.fashion_mnist(),
    ]
    datasets = ["MNIST", "Fashion MNIST", "KMNIST"]
    epochs = np.arange(1, NUM_EPOCHS + 1)

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
    axes[0].set_xlim([0, NUM_EPOCHS])

    utils.save_fig(fig, "training")
