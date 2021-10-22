import tensorflow as tf


def ideal_I(V, G):
    """Computes output currents of an ideal crossbar.

    Parameters
    ----------
    V : ndarray
        Voltages of shape `p x m`.
    G : ndarray
        Conductances of shape `m x n`.

    Returns
    ----------
    ndarray
        Output currents of shape `p x n`.
    """
    return tf.tensordot(V, G, axes=1)