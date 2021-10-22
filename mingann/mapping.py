import tensorflow as tf


def G_min_and_G_max(conductance_levels):
    return conductance_levels[0], conductance_levels[-1]


def I_to_y(I, k_V, max_weight, conductance_levels):
    G_min, G_max = G_min_and_G_max(conductance_levels)

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
    G_min, G_max = G_min_and_G_max(conductance_levels)

    max_weight = tf.math.reduce_max(tf.math.abs(weights))

    k_G = compute_k_G(max_weight, G_max, G_min)
    G_eff = k_G*weights

    # We implement the pairs by choosing the lowest possible conductances.
    G_pos = tf.math.maximum(G_eff, 0.0) + G_min
    G_neg = -tf.math.minimum(G_eff, 0.0) + G_min

    # Odd columns dedicated to positive weights.
    # Even columns dedicated to positive weights.
    G = tf.reshape(
            tf.concat([G_pos[..., tf.newaxis], G_neg[..., tf.newaxis]], axis=-1),
            [tf.shape(G_pos)[0], -1]
            )


    G = round_to_closest(G, conductance_levels)

    return G, max_weight


def round_to_closest(array, values):
    """Round elements in `array` to the closest elements in `values`."""
    array_1d = tf.reshape(array, [-1])
    idx = tf.searchsorted(values, array_1d, side="left")
    idx = idx - tf.where((tf.math.abs(array_1d - tf.gather(values, idx-1)) < tf.math.abs(array_1d - tf.gather(values, idx))), 1, 0)
    array_1d = tf.gather(values, idx)
    array = tf.reshape(array_1d, array.shape)
    return array
