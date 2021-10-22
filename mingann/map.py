import numpy as np


def I_to_y(I, k_V, max_weight, G_max, G_min):
    I_total = I[:, 0::2] - I[:, 1::2]
    y = I_total_to_y(I_total, k_V, max_weight, G_max, G_min)
    return y


def I_total_to_y(I_total, k_V, max_weight, G_max, G_min):
    k_G = compute_k_G(max_weight, G_max, G_min)
    k_I = compute_k_I(k_V, k_G)
    y = I_total/k_I
    return y


def compute_k_G(max_weight, G_max, G_min):
    k_G = (G_max-G_min)/max_weight

    return k_G


def compute_k_I(k_V, k_G):
    return k_V*k_G


def x_to_V(x, k_V):
    return k_V*x


def w_to_G(weights, conductance_levels):
    G_min = conductance_levels[0]
    G_max = conductance_levels[-1]

    max_weight = np.max(np.abs(weights))

    k_G = compute_k_G(max_weight, G_max, G_min)
    G_eff = k_G*weights

    # We implement the pairs by choosing the lowest possible conductances.
    G_pos = np.maximum(G_eff, 0.0) + G_min
    G_neg = -np.minimum(G_eff, 0.0) + G_min

    G = np.zeros((weights.shape[0], 2*weights.shape[1]))
    # Odd columns dedicated to positive weights.
    G[:, ::2] = G_pos
    # Even columns dedicated to positive weights.
    G[:, 1::2] = G_neg

    G = round_to_closest(G, conductance_levels)

    return G, max_weight


def round_to_closest(array, values):
    """Round elements in `array` to the closest elements in `values`."""
    array_1d = array.flatten()
    idx = np.searchsorted(values, array_1d, side="left")
    idx = idx - (np.abs(array_1d - values[idx-1]) < np.abs(array_1d - values[idx]))
    array_1d = values[idx]
    array = np.reshape(array_1d, array.shape)
    return array
