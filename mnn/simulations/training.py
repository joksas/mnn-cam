from mnn.network import config


def mnist():
    return config.TrainingConfig("mnist", 32, 10)


def fashion_mnist():
    return config.TrainingConfig("fashion_mnist", 32, 10)


def kmnist():
    return config.TrainingConfig("kmnist", 32, 10)
