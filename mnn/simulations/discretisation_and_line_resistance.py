from mnn import crossbar, expdata
from mnn.network import config

from . import training


def configs():
    levels = [32, 128, 303, 370, 526]
    filenames = [f"{n}-levels-retention.xlsx" for n in levels]
    conductances_levels_lst = [
        expdata.load.retention_conductance_levels(filename) for filename in filenames
    ]
    nonidealities = [
        crossbar.nonidealities.Discretised(conductance_levels)
        for conductance_levels in conductances_levels_lst
    ]
    Gs = [
        {"G_off": conductance_levels[0], "G_on": conductance_levels[-1]}
        for conductance_levels in conductances_levels_lst
    ]

    line_resistance_nonideality = crossbar.nonidealities.LineResistance(128, 64, 0.35, 0.32)


    training_setups = [training.mnist(), training.fashion_mnist()]

    config_lst = []

    for training_setup in training_setups:
        config_lst.append(config.InferenceConfig(training_setup, None, 1))
        for nonideality, G in zip(nonidealities, Gs):
            config_lst.append(config.InferenceConfig(training_setup, [nonideality, line_resistance_nonideality], 1, **G))

    return config_lst


def run():
    training.mnist().run()
    training.fashion_mnist().run()

    for c in configs():
        c.run()
