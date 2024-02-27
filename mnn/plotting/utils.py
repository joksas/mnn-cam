import logging
import os


def color_list() -> list[str]:
    """Return colors of Okabe-Ito colorblind-friendly palette.

    Returns:
        HEX color codes.
    """
    colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    return colors


def color_dict() -> dict[str, str]:
    """Return same as `colors_list()` but dict."""
    color_names = [
        "orange",
        "sky-blue",
        "bluish-green",
        "yellow",
        "blue",
        "vermilion",
        "reddish-purple",
        "black",
    ]
    colors = dict(zip(color_names, color_list()))
    return colors


def save_fig(fig, name: str):
    dir_name = "plots"
    os.makedirs(dir_name, exist_ok=True)
    path = os.path.join(dir_name, f"{name}.pdf")
    # if os.path.exists(path):
    #     logging.warning(f'File "{path}" already exists. Skipping...')
    #     return

    fig.savefig(path, bbox_inches="tight", transparent=True)
    logging.info(f'Saved file "{path}".')


def add_boxplot_legend(axis, boxplots, labels, linewdith=1.0, loc="upper right"):
    leg = axis.legend(
        [boxplot["boxes"][0] for boxplot in boxplots],
        labels,
        fontsize=8,
        frameon=False,
        loc=loc,
    )
    for line in leg.get_lines():
        line.set_linewidth(linewdith)
