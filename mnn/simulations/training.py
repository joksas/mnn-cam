from mnn.network import config


def mnist():
    return config.TrainingConfig("mnist", 32, 1)


def fashion_mnist():
    return config.TrainingConfig("fashion_mnist", 32, 1)
