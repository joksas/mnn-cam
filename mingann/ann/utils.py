import numpy as np


def save_numpy(file_path: str, data: np.ndarray):
    with open(file_path, "wb") as file:
        np.save(file, data)