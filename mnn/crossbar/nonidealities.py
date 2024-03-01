from abc import ABC, abstractmethod

import badcrossbar
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class Nonideality(ABC):
    """Physical effect that influences the behavior of memristive devices."""

    @abstractmethod
    def label(self) -> str:
        """Returns nonideality label used in directory names, for example."""

    def __eq__(self, other):
        if self is None or other is None:
            if self is None and other is None:
                return True
            return False
        return self.label() == other.label()


class LinearityPreserving(ABC):
    """Nonideality whose effect can be simulated by disturbing the conductances."""

    @abstractmethod
    def disturb_G(self, G: tf.Tensor) -> tf.Tensor:
        """Disturb conductances."""


class LinearityNonpreserving(ABC):
    @abstractmethod
    def compute_I(self, V: tf.Tensor, G: tf.Tensor) -> tf.Tensor:
        """Compute currents in a crossbar suffering from linearity-nonpreserving nonideality.

        Args:
            V: Voltages of shape `p x m`.
            G: Conductances of shape `m x n`.

        Returns:
            I: Output currents of shape `p x n`.
                conductances in the crossbar array.
        """

    @abstractmethod
    def k_V(self) -> float:
        """Return voltage scaling factor."""


class Discretised(Nonideality, LinearityPreserving):
    """Assumes that devices can be set to only a finite number of conductance states."""

    def __init__(self, G_levels) -> None:
        """
        Args:
            G_levels: Available conductance levels.
        """
        self.G_levels = G_levels

    def label(self):
        G_off = self.G_levels[0]
        G_on = self.G_levels[-1]
        return f"discretised={{num_states={len(self.G_levels)},G_off={G_off:.3g},G_on={G_on:.3g}}}"

    def disturb_G(self, G):
        G_1d = tf.reshape(G, [-1])
        # It seems that previous operations may introduce small deviations outside the original range.
        G_1d = tf.clip_by_value(G_1d, self.G_levels[0], self.G_levels[-1])
        idx = tf.searchsorted(self.G_levels, G_1d, side="left")
        idx = idx - tf.where(
            (
                tf.math.abs(G_1d - tf.gather(self.G_levels, tf.math.maximum(0, idx - 1)))
                < tf.math.abs(G_1d - tf.gather(self.G_levels, idx))
            ),
            1,
            0,
        )
        G_1d = tf.gather(self.G_levels, idx)
        G = tf.reshape(G_1d, G.shape)
        return G

class LognormalWithTrend(Nonideality, LinearityPreserving):
    """Assumes that resistance states are lognormally distributed when programmed. Lognormal parameters are determined by a trend line."""

    def __init__(self, mu_trend: tuple[float, float], sigma_trend: tuple[float, float]) -> None:
        """
        Args:
            mu_trend: slope and intercept of the trend line of ln(R) for the mu parameter.
            sigma_trend: slope and intercept of the trend line of ln(R) for the sigma parameter.
        """
        self.mu_trend = mu_trend
        self.sigma_trend = sigma_trend

    def label(self):
        return f"lognormal-with-trend={{mu_trend={self.mu_trend},sigma_trend={self.sigma_trend}}}"

    def disturb_G(self, G):
        # Handle G = 0 case.
        R = tf.where(G == 0, 1e-12, 1 / G)

        mus = self.mu_trend[0] * tf.math.log(R) + self.mu_trend[1]
        sigmas = self.sigma_trend[0] * tf.math.log(R) + self.sigma_trend[1]
        sigmas = tf.where(sigmas <= 0, 1e-12, sigmas)

        distribution = tfd.LogNormal(mus, sigmas)
        new_R = distribution.sample(shape=R.shape)

        return 1 / new_R


class LineResistance(Nonideality, LinearityNonpreserving):
    """Takes interconnect resistance into account."""

    def __init__(self, num_word_lines: int, num_bit_lines: int, word_line_r: float, bit_line_r: float) -> None:
        """
        Args:
            num_word_lines: Number of word lines.
            num_bit_lines: Number of bit lines.
            word_line_r: Interconnect resistance (in ohms) in word lines.
            bit_line_r: Interconnect resistance (in ohms) in bit lines.
        """
        self.num_word_lines = num_word_lines
        self.num_bit_lines = num_bit_lines
        self.word_line_r = word_line_r
        self.bit_line_r = bit_line_r

    def k_V(self):
        return 0.5

    def label(self):
        return f"line-resistance={{num_word_lines={self.num_word_lines},num_bit_lines={self.num_bit_lines},word_line_r={self.word_line_r:.3g},bit_line_r={self.bit_line_r:.3g}}}"

    def compute_I(self, V, G):
        R = tf.where(G == 0, 1e-12, 1 / G)

        return line_resistance(R, V, self.num_word_lines, self.num_bit_lines, self.word_line_r, self.bit_line_r)

def line_resistance(R: tf.Tensor, V: tf.Tensor, num_word_lines: int, num_bit_lines: int, word_line_r: float, bit_line_r: float) -> tf.Tensor:
    """Takes interconnect resistance into account.

    Args:
        R: Resistances of shape `p x n`.
        V: Voltages of shape `p x m`.
        num_word_lines: Number of word lines.
        num_bit_lines: Number of bit lines.
        word_line_r: Interconnect resistance (in ohms) in word lines.
        bit_line_r: Interconnect resistance (in ohms) in bit lines.

    Returns:
        Output currents of shape `p x n`.
    """
    crossbar_resistances = map_resistances_to_crossbars(R, num_word_lines, num_bit_lines)
    crossbar_voltages = map_voltages_to_crossbars(tf.transpose(V), num_word_lines)

    output_currents_list = []

    for i in range(crossbar_resistances.shape[0]):
        voltages = crossbar_voltages[i].numpy()
        for j in range(crossbar_resistances.shape[1]):
            resistances = crossbar_resistances[i, j].numpy()
            output_currents = badcrossbar.compute(voltages, resistances, r_i_word_line=word_line_r, r_i_bit_line=bit_line_r).currents.output
            output_currents_list.append(output_currents.astype(np.float32))

    return combine_currents_from_multiple_crossbars(output_currents_list)


def map_resistances_to_crossbars(R: tf.Tensor, num_word_lines: int, num_bit_lines: int) -> tf.Tensor:
    """Maps resistances to crossbars. If the shape doesn't match, infinite resistances are appended.

    Args:
        R: Resistances of shape `p x n`.
        num_word_lines: Number of word lines.
        num_bit_lines: Number of bit lines.

    Returns:
        Crossbars in shape `num_vertical_multiplier x num_horizontal_multiplier x num_word_lines x num_bit_lines`.
    """
    num_horizontal_multiplier = R.shape[1] // num_bit_lines
    num_vertical_multiplier = R.shape[0] // num_word_lines

    if R.shape[0] % num_word_lines != 0:
        R = tf.concat([R, tf.fill([num_word_lines - R.shape[0] % num_word_lines, R.shape[1]], tf.constant(np.inf, dtype=R.dtype))], axis=0)
        num_vertical_multiplier += 1
    if R.shape[1] % num_bit_lines != 0:
        R = tf.concat([R, tf.fill([R.shape[0], num_bit_lines - R.shape[1] % num_bit_lines], tf.constant(np.inf, dtype=R.dtype))], axis=1)
        num_horizontal_multiplier += 1

    crossbar_resistances = tf.reshape(R, [num_vertical_multiplier, num_horizontal_multiplier, num_word_lines, num_bit_lines])

    return crossbar_resistances

def map_voltages_to_crossbars(V: tf.Tensor, num_word_lines: int) -> tf.Tensor:
    """Maps voltages to crossbars. If the shape doesn't match, zero voltages are appended.

    Args:
        V: Voltages of shape `p x m`.
        num_word_lines: Number of word lines.
        num_bit_lines: Number of bit lines.

    Returns:
    """
    num_vertical_multiplier = V.shape[0] // num_word_lines

    if V.shape[0] % num_word_lines != 0:
        V = tf.concat([V, tf.fill([num_word_lines - V.shape[0] % num_word_lines, tf.shape(V)[1]], tf.constant(0, dtype=V.dtype))], axis=0)
        num_vertical_multiplier += 1

    crossbar_voltages = tf.reshape(V, [num_vertical_multiplier, num_word_lines, -1])

    return crossbar_voltages

def combine_currents_from_multiple_crossbars(output_currents: list[tf.Tensor]) -> tf.Tensor:
    """Combines currents from multiple crossbars.

    Args:
        output_currents: List of output currents of shape `p x n`.

    Returns:
        Combined currents of shape `p x n`.
    """
    return tf.reduce_sum(output_currents, axis=0)
