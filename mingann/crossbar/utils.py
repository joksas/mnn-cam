import tensorflow as tf


def add_I_BL(I_ind: tf.Tensor) -> tf.Tensor:
    """Add currents along the bit lines.

    Args:
        I_ind: Currents of shape `p x m x n` produced by each of the conductances in the crossbar
            array.

    Returns:
        Output currents of shape `p x n`.
    """
    I = tf.math.reduce_sum(I_ind, axis=1)
    return I
