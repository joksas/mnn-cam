import tensorflow as tf
from tensorflow.keras import layers, models

from mnn import crossbar

from . import utils


def get_model(
    dataset,
    custom_weights_path=None,
    memristive_config=None,
):
    model = models.Sequential()

    if dataset in ["mnist", "fashion_mnist", "kmnist"]:
        model.add(layers.Flatten(input_shape=(28, 28)))
        model.add(MemristorDense(25, memristive_config=memristive_config))
        model.add(layers.Activation("sigmoid"))
        model.add(MemristorDense(10, memristive_config=memristive_config))
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
    def __init__(self, units, memristive_config=None, **kwargs):
        layers.Dense.__init__(self, units, **kwargs)
        self.memristive_config = memristive_config

    def call(self, inputs):
        if self.memristive_config is not None:
            return self.memristive_outputs(inputs)
        return self.standard_outputs(inputs)

    def combined_weights(self):
        return tf.concat([self.kernel, tf.expand_dims(self.bias, axis=0)], 0)

    def combined_inputs(self, inputs):
        ones = tf.ones([tf.shape(inputs)[0], 1])
        inputs = tf.concat([inputs, ones], 1)
        return inputs

    def standard_outputs(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

    def memristive_outputs(self, x):
        k_V = self.memristive_config["k_V"]
        G_off = self.memristive_config["G_off"]
        G_on = self.memristive_config["G_on"]
        mapping_rule = self.memristive_config["mapping_rule"]
        nonidealities = self.memristive_config["nonidealities"]
        power_path = self.memristive_config["power_path"]

        # Mapping inputs onto voltages.
        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)
        V = crossbar.map.x_to_V(inputs, k_V)

        # Mapping weights onto conductances.
        G, max_weight = crossbar.map.w_to_G(
            self.combined_weights(), G_off, G_on, mapping_rule=mapping_rule
        )

        # Linearity-preserving nonidealities
        for nonideality in nonidealities:
            if isinstance(nonideality, crossbar.nonidealities.LinearityPreserving):
                G = nonideality.disturb_G(G)

        # Linearity-nonpreserving nonidealities
        I = None
        I_ind = None
        for nonideality in nonidealities:
            if isinstance(nonideality, crossbar.nonidealities.LinearityNonpreserving):
                I, I_ind = nonideality.compute_I(V, G)

        # Ideal case for computing output currents.
        if I is None or I_ind is None:
            I, I_ind = crossbar.ideal.compute_I_all(V, G)

        if power_path is not None:
            P_avg = utils.compute_avg_crossbar_power(V, I_ind)
            with open(power_path, mode="a", encoding="utf-8"):
                tf.print(P_avg, output_stream=f"file://{power_path}")

        # Converting to outputs.
        y_disturbed = crossbar.map.I_to_y(I, k_V, max_weight, G_on, G_off)

        return y_disturbed
