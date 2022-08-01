from mnn import crossbar, expdata
from mnn.network import config

from . import training


def run():
    mnist_training = training.mnist()
    mnist_training.run()

    conductance_levels = expdata.load.retention_conductance_levels("32-levels-retention.xlsx")
    G_off, G_on = conductance_levels[0], conductance_levels[-1]

    ideal_inference = config.InferenceConfig(mnist_training, [], 1, G_off=G_off, G_on=G_on)
    ideal_inference.run()

    nonideality = crossbar.nonidealities.Discretised(conductance_levels)
    nonideal_inference = config.InferenceConfig(
        mnist_training, [nonideality], 1, G_off=G_off, G_on=G_on
    )
    nonideal_inference.run()
