from mnn import crossbar, expdata
from mnn.network import config


def get_config():
    conductance_levels = expdata.load.retention_conductance_levels("32-levels-retention.xlsx")
    G_off, G_on = conductance_levels[0], conductance_levels[-1]
    nonideality = crossbar.nonidealities.Discretised(conductance_levels)

    return config.SimulationConfig(
        "mnist",
        config.TrainingConfig(32, 10, num_epochs=1000),
        config.InferenceConfig(1),
        G_off=G_off,
        G_on=G_on,
        nonidealities=[nonideality],
    )


def run():
    simulation_config = get_config()
    simulation_config.train()
    simulation_config.infer()
