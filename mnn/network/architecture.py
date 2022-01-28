import tensorflow as tf
from tensorflow.keras import layers, models

from mnn import crossbar, expdata

from . import utils


def get_model(dataset, custom_weights_path=None, is_memristive=True, power_path=None):
    conductance_levels = expdata.load.retention_conductance_levels()

    model = models.Sequential()

    if dataset == "mnist":
        model.add(layers.Flatten(input_shape=(28, 28)))
        model.add(MemristorDense(25, conductance_levels, is_memristive, power_path=power_path))
        model.add(layers.Activation("sigmoid"))
        model.add(MemristorDense(10, conductance_levels, is_memristive, power_path=power_path))
        model.add(layers.Activation("softmax"))
    else:
        raise ValueError('Dataset "{dataset}" is not supported.')

    if custom_weights_path is not None:
        model.load_weights(custom_weights_path)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


class MemristorDense(layers.Dense):
    def __init__(self, units, conductance_levels, is_memristive, power_path=None, **kwargs):
        self.is_memristive = is_memristive
        self.conductance_levels = conductance_levels
        self.power_path = power_path
        layers.Dense.__init__(self, units, **kwargs)

    def call(self, inputs):
        if self.is_memristive:
            return self.memristive_outputs(inputs)
        else:
            return self.standard_outputs(inputs)

    def combined_weights(self):
        return tf.concat([self.kernel, tf.expand_dims(self.bias, axis=0)], 0)

    def combined_inputs(self, inputs):
        ones = tf.ones([tf.shape(inputs)[0], 1])
        inputs = tf.concat([inputs, ones], 1)
        return inputs

    def standard_outputs(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

    def memristive_outputs(self, inputs):
        weights = self.combined_weights()
        inputs = self.combined_inputs(inputs)

        # Mapping inputs onto voltages.
        k_V = 0.5
        V = crossbar.map.x_to_V(inputs, k_V)

        # Mapping weights onto conductances.
        G, max_weight = crossbar.map.w_to_G(weights, self.conductance_levels)

        # Ideal case for computing output currents.
        I, I_ind = crossbar.ideal.compute_I_all(V, G)

        if self.power_path is not None:
            P_avg = utils.compute_avg_crossbar_power(V, I_ind)
            with open(self.power_path, mode="a", encoding="utf-8"):
                tf.print(P_avg, output_stream=f"file://{self.power_path}")

        y_disturbed = crossbar.map.I_to_y(I, k_V, max_weight, self.conductance_levels)

        return y_disturbed
