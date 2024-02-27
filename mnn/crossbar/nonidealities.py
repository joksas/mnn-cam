from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from scipy.stats import lognorm
from tensorflow_probability import distributions as tfd

tf.config.run_functions_eagerly(True)


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
    """Nonideality in which nonlinearity manifests itself in individual devices
    and the output current of a device is a function of its conductance
    parameter and the voltage applied across it."""

    @abstractmethod
    def compute_I(self, V: tf.Tensor, G: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute currents in a crossbar suffering from linearity-nonpreserving nonideality.

        Args:
            V: Voltages of shape `p x m`.
            G: Conductances of shape `m x n`.

        Returns:
            I: Output currents of shape `p x n`.
            I_ind: Currents of shape `p x m x n` produced by each of the
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
