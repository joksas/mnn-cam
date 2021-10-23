from mingann.ann import config


def get_config():
    return config.SimulationConfig(
        "mnist",
        config.TrainingConfig(32, 10, num_epochs=1000),
        config.InferenceConfig(1),
    )


def run():
    simulation_config = get_config()
    simulation_config.train()
    simulation_config.infer()