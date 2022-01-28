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


def get_energy_efficiency(
    avg_power: float,
    num_neurons_lst: list[int] = [784, 25, 10],
    read_time: float = 50e-9,
):
    num_synapses = get_num_synapses(num_neurons_lst)
    energy_efficiency = (2 * num_synapses) / (read_time * avg_power)
    return energy_efficiency


def get_num_synapses(num_neurons_lst: list[int]):
    num_synapses = 0
    for idx, num_neurons in enumerate(num_neurons_lst[:-1]):
        num_synapses += (num_neurons + 1) * num_neurons_lst[idx + 1]

    return num_synapses
