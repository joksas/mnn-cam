import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_retention():
    path = os.path.join(Path(__file__).parent.parent.absolute(), "expdata", "32-levels-retention.xlsx")
    whole_table = pd.read_excel(path)
    retention_data = np.array(whole_table.iloc[:1000, 3:])
    return retention_data
