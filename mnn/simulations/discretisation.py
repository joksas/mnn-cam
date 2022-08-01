from mnn import crossbar, expdata
from mnn.network import config


def run():
    conductance_levels = expdata.load.retention_conductance_levels("32-levels-retention.xlsx")
    G_off, G_on = conductance_levels[0], conductance_levels[-1]
    nonideality = crossbar.nonidealities.Discretised(conductance_levels)

    training = config.TrainingConfig("mnist", 32, 1)
    training.run()

    ideal_inference = config.InferenceConfig(training, [], 1, G_off=G_off, G_on=G_on)
    ideal_inference.run()

    nonideal_inference = config.InferenceConfig(training, [nonideality], 1, G_off=G_off, G_on=G_on)
    nonideal_inference.run()
