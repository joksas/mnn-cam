import tensorflow as tf


def assert_tf_approx(a, b):
    a = tf.where(tf.math.is_inf(a), tf.zeros_like(a), a)
    b = tf.where(tf.math.is_inf(b), tf.zeros_like(b), b)
    tf.debugging.assert_near(a, b, rtol=1.0e-6, atol=1.0e-6)
    assert a.shape == b.shape
