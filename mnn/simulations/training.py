from mnn.network import config


def mnist():
    return config.TrainingConfig("mnist", 32, 10)
