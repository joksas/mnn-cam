import tensorflow as tf
from tensorflow.keras import layers, models
from mingann import crossbar


class MemristorDense(layers.Dense):
    def __init__(self, units, conductance_levels, **kwargs):
        self.is_memristive = True
        self.conductance_levels = conductance_levels
        layers.Dense.__init__(self, units, **kwargs)

    def call(self, inputs):
        if self.is_memristive:
            return self.memristive_outputs(inputs)
        else:
            return self.standard_outputs(inputs)

    def combined_weights(self):
        return tf.concat([self.kernel, tf.expand_dims(self.bias, axis=0)], 0)

    def standard_outputs(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

    def memristive_outputs(self, inputs):
        weights = self.combined_weights()

        # Mapping inputs onto voltages.
        k_V = 1.0
        V = crossbar.map.x_to_V(inputs, k_V)

        # Mapping weights onto conductances.
        G, max_weight = crossbar.map.w_to_G(weights, self.conductance_levels)

        # Ideal case for computing output currents.
        I = crossbar.compute.ideal_I(V, G)

        y_disturbed = crossbar.map.I_to_y(I, k_V, max_weight, self.conductance_levels)

        return y_disturbed
