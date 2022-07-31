from abc import ABC, abstractmethod

import tensorflow as tf


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
        return f"discretised:{len(self.G_levels)}"

    def disturb_G(self, G):
        G_1d = tf.reshape(G, [-1])
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
