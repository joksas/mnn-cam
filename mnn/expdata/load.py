import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import lognorm


def retention_table(filename):
    path = os.path.join(Path(__file__).parent.parent.parent.absolute(), "expdata", filename)
    whole_table = pd.read_excel(path).dropna()
    return whole_table


def retention_currents(filename):
    currents = retention_table(filename).iloc[:, 1:]
    currents = tf.constant(currents, dtype=tf.float32)
    return currents


def retention_voltages(filename):
    voltages = retention_table(filename).iloc[:, 0]
    voltages = tf.constant(voltages, dtype=tf.float32)
    return voltages


def retention_conductances(filename):
    voltages = retention_voltages(filename)
    abs_voltages = tf.math.abs(voltages)
    currents = retention_currents(filename)
    conductances = currents / abs_voltages[:, None]
    return conductances


def retention_conductance_levels(filename):
    avg_levels = tf.math.reduce_mean(retention_conductances(filename), axis=0)
    return tf.sort(avg_levels)

def c2c_data() -> list[tuple[float, float, float]]:
    """
    Return average voltage, lognormal sigma, and lognormal mu for each resistance state.
    """
    path = os.path.join(Path(__file__).parent.parent.parent.absolute(), "expdata", "uniformity-data.xlsx")
    data = pd.read_excel(path, sheet_name=None)
    lst = []

    for _, df in data.items():
        df = df[(-0.507 <= df["CH1 Source"]) & (df["CH1 Source"] <= -0.505)]

        low_R_states = df.iloc[::2, :]
        resistances = np.abs(low_R_states["CH1 Source"].values / low_R_states["CH1 Current"].values)
        avg_voltage = np.abs(low_R_states["CH1 Source"].mean())
        avg_current = np.abs(low_R_states["CH1 Current"].mean())
        fit_params = lognorm.fit(resistances, floc=0)
        sigma = fit_params[0]
        mu = np.log(fit_params[2])
        lst.append((avg_voltage/avg_current, mu, sigma))

        high_R_states = df.iloc[1::2, :]
        resistances = np.abs(high_R_states["CH1 Source"].values / high_R_states["CH1 Current"].values)
        avg_voltage = np.abs(high_R_states["CH1 Source"].mean())
        avg_current = np.abs(high_R_states["CH1 Current"].mean())
        fit_params = lognorm.fit(resistances, floc=0)
        sigma = fit_params[0]
        mu = np.log(fit_params[2])
        lst.append((avg_voltage/avg_current, mu, sigma))

    return lst
