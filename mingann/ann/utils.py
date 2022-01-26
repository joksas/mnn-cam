import numpy as np
import tensorflow as tf


def save_numpy(file_path: str, data: np.ndarray):
    with open(file_path, "wb") as file:
        np.save(file, data)


def _compute_device_power(V: tf.Tensor, I_ind: tf.Tensor) -> tf.Tensor:
    """Compute power dissipated by individual devices in a crossbar.

    Args:
        V: Voltages in shape `p x m` with `p` examples applied across `m` word lines.
        I_ind: Currents in shape `p x m x n` generated by the individual devices in crossbar with
            `m` word lines and `n` bit lines.

    Returns:
        Power in shape `p x m x n`.
    """
    # $P = |V| |I|$ for individual devices. All devices in the same word line
    # of the crossbar (row of G) are applied with the same voltage.
    P_ind = tf.einsum("ij,ijk->ijk", tf.math.abs(V), tf.math.abs(I_ind))

    return P_ind


def compute_avg_crossbar_power(V: tf.constant, I_ind: tf.constant) -> float:
    """Compute average power dissipated by a crossbar.

    Args:
        V: Voltages in shape `p x m` with `p` examples applied across `m` word lines.
        I_ind: Currents in shape `p x m x n` generated by the individual devices in crossbar with
            `m` word lines and `n` bit lines.

    Returns:
        Average power dissipated by a crossbar.
    """
    P = _compute_device_power(V, I_ind)
    P_sum = tf.math.reduce_sum(P)
    # To get average power consumption **per crossbar** we divide by number of examples.
    P_avg = P_sum / tf.cast(tf.shape(V)[0], tf.float32)

    return P_avg
