import os
from pathlib import Path

import pandas as pd
import tensorflow as tf


def retention_table():
    path = os.path.join(
        Path(__file__).parent.parent.parent.absolute(), "expdata", "32-levels-retention.xlsx"
    )
    whole_table = pd.read_excel(path)
    return whole_table


def retention_currents():
    currents = retention_table().iloc[:1000, 3:]
    currents = tf.constant(currents, dtype=tf.float32)
    return currents


def retention_voltages():
    voltages = retention_table().iloc[:1000, 1]
    voltages = tf.constant(voltages, dtype=tf.float32)
    return voltages


def retention_conductances():
    voltages = retention_voltages()
    abs_voltages = tf.math.abs(voltages)
    currents = retention_currents()
    conductances = currents / abs_voltages[:, None]
    return conductances


def retention_conductance_levels():
    avg_levels = tf.math.reduce_mean(retention_conductances(), axis=0)
    return tf.sort(avg_levels)
