from mnn import crossbar, expdata
from mnn.network import config

from . import training


def configs():
    conductance_levels_32 = expdata.load.retention_conductance_levels("32-levels-retention.xlsx")
    nonideality_32 = crossbar.nonidealities.Discretised(conductance_levels_32)
    G_32 = {"G_off": conductance_levels_32[0], "G_on": conductance_levels_32[-1]}

    conductance_levels_370 = expdata.load.retention_conductance_levels("370-levels-retention.xlsx")
    nonideality_370 = crossbar.nonidealities.Discretised(conductance_levels_370)
    G_370 = {"G_off": conductance_levels_370[0], "G_on": conductance_levels_370[-1]}

    return [
        # MNIST
        config.InferenceConfig(training.mnist(), None, 1),
        config.InferenceConfig(training.mnist(), [nonideality_32], 1, **G_32),
        config.InferenceConfig(training.mnist(), [nonideality_370], 1, **G_370),
        # Fashion MNIST
        config.InferenceConfig(training.fashion_mnist(), None, 1),
        config.InferenceConfig(training.fashion_mnist(), [nonideality_32], 1, **G_32),
        config.InferenceConfig(training.fashion_mnist(), [nonideality_370], 1, **G_370),
    ]


def run():
    training.mnist().run()
    training.fashion_mnist().run()

    for c in configs():
        c.run()
