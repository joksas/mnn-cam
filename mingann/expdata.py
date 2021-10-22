import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_retention_table():
    path = os.path.join(Path(__file__).parent.parent.absolute(), "expdata", "32-levels-retention.xlsx")
    whole_table = pd.read_excel(path)
    return whole_table


def load_retention_currents():
    retention_currents = load_retention_table().iloc[:1000, 3:]
    retention_currents = np.array(retention_currents)
    return retention_currents


def load_retention_voltages():
    retention_voltages = load_retention_table().iloc[:1000, 1]
    retention_voltages = np.array(retention_voltages)
    return retention_voltages
