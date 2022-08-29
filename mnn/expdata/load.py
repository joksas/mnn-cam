import os
from pathlib import Path

import pandas as pd
import tensorflow as tf


def _full_filepath(filename):
    return os.path.join(Path(__file__).parent.parent.parent.absolute(), "expdata", filename)


def retention_table(filename):
    path = _full_filepath(filename)
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


def sweeping_data(filename):
    path = _full_filepath(filename)
    whole_table = pd.read_excel(path)
    sweeps = []
    for col_pair_idx in range(whole_table.shape[1] // 2):
        sweeps.append(
                tf.constant(
                    whole_table.iloc[:, [2*col_pair_idx, 2*col_pair_idx+1]].dropna(), dtype=tf.float32
                    )
        )

    return sweeps
