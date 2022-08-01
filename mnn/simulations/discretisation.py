from mnn import crossbar, expdata
from mnn.network import config

from . import training

conductance_levels = expdata.load.retention_conductance_levels("32-levels-retention.xlsx")
G_off, G_on = conductance_levels[0], conductance_levels[-1]


def nonideal_config():
    nonideality = crossbar.nonidealities.Discretised(conductance_levels)
    return config.InferenceConfig(training.mnist(), [nonideality], 1, G_off=G_off, G_on=G_on)


def ideal_config():
    return config.InferenceConfig(training.mnist(), [], 1, G_off=G_off, G_on=G_on)


def run():
    training.mnist().run()
    ideal_config().run()
    nonideal_config.run()
