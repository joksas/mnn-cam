import tensorflow as tf
from scipy.optimize import curve_fit

from mnn import crossbar, expdata
from mnn.network import config

from . import training


def fit_log_x_to_y(x, y):
    popt, _ = curve_fit(lambda x, a, b: a * tf.math.log(x) + b, x, y)
    return popt


def configs():
    levels = [32, 128, 303, 370, 526]
    filenames = [f"{n}-levels-retention.xlsx" for n in levels]
    conductances_levels_lst = [
        expdata.load.retention_conductance_levels(filename) for filename in filenames
    ]

    c2c_data = expdata.load.c2c_data()
    mu_fit_params = fit_log_x_to_y([c2c_point[0] for c2c_point in c2c_data], [data[1] for data in c2c_data])
    sigma_fit_params = fit_log_x_to_y([c2c_point[0] for c2c_point in c2c_data], [data[2] for data in c2c_data])
    d2d_nonideality = crossbar.nonidealities.LognormalWithTrend(mu_fit_params, sigma_fit_params)

    discretized_nonidealities = [
        crossbar.nonidealities.Discretised(conductance_levels)
        for conductance_levels in conductances_levels_lst
    ]
    Gs = [
        {"G_off": conductance_levels[0], "G_on": conductance_levels[-1]}
        for conductance_levels in conductances_levels_lst
    ]

    training_setups = [training.mnist(), training.fashion_mnist()]

    config_lst = []

    for training_setup in training_setups:
        config_lst.append(config.InferenceConfig(training_setup, None, 1))
        for discretized_nonideality, G in zip(discretized_nonidealities, Gs):
            config_lst.append(config.InferenceConfig(training_setup, [discretized_nonideality, d2d_nonideality], 5, **G))

    return config_lst


def run():
    training.mnist().run()
    training.fashion_mnist().run()

    for c in configs():
        c.run()

