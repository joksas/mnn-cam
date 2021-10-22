import pandas as pd
import os
from pathlib import Path
import tensorflow as tf


def load_retention_table():
    path = os.path.join(Path(__file__).parent.parent.absolute(), "expdata", "32-levels-retention.xlsx")
    whole_table = pd.read_excel(path)
    return whole_table


def load_retention_currents():
    retention_currents = load_retention_table().iloc[:1000, 3:]
    retention_currents = tf.constant(retention_currents, dtype=tf.float32)
    return retention_currents


def load_retention_voltages():
    retention_voltages = load_retention_table().iloc[:1000, 1]
    retention_voltages = tf.constant(retention_voltages, dtype=tf.float32)
    return retention_voltages


def load_retention_conductances():
    retention_voltages = load_retention_voltages()
    abs_retention_voltages = tf.math.abs(retention_voltages)
    retention_currents = load_retention_currents()
    retention_conductances  = retention_currents/abs_retention_voltages[:, None]
    return retention_conductances


def load_retention_conductance_levels():
    avg_levels = tf.math.reduce_mean(load_retention_conductances(), axis=0)
    return tf.sort(avg_levels)
