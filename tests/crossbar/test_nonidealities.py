import pytest
import tensorflow as tf

from mnn import crossbar
from tests import utils

discretised_testdata = [
    (
        {
            "weights": tf.constant(
                [
                    [4.0, 2.0, -5.0],
                    [-1.0, 0.0, 1.0],
                ]
            ),
            "conductance_levels": tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        },
        [
            tf.constant(
                [
                    [5.0, 1.0, 3.0, 1.0, 1.0, 6.0],
                    [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
                ]
            ),
            5.0,
        ],
    ),
    (
        {
            "weights": tf.constant(
                [
                    [4.2, 2.1, -5.0],
                    [-1.0, 0.9, 0.2],
                ]
            ),
            "conductance_levels": tf.constant([1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5, 5.0e-5, 6.0e-5]),
        },
        [
            tf.constant(
                [
                    [5.0e-5, 1.0e-5, 3.0e-5, 1.0e-5, 1.0e-5, 6.0e-5],
                    [1.0e-5, 2.0e-5, 2.0e-5, 1.0e-5, 1.0e-5, 1.0e-5],
                ]
            ),
            5.0,
        ],
    ),
    (
        {
            "weights": tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [-1.0, -2.0, 3.0],
                ]
            ),
            "conductance_levels": tf.constant([1.1, 2.0, 2.1]),
        },
        [
            tf.constant(
                [
                    [1.1, 1.1, 2.0, 1.1, 2.1, 1.1],
                    [1.1, 1.1, 1.1, 2.0, 2.1, 1.1],
                ]
            ),
            3.0,
        ],
    ),
]


@pytest.mark.parametrize("args,expected", discretised_testdata)
def test_w_to_G(args, expected):
    G_off, G_on = args["conductance_levels"][0], args["conductance_levels"][-1]
    G_exp, max_weight_exp = expected
    G, max_weight = crossbar.map.w_to_G(args["weights"], G_off, G_on)
    nonideality = crossbar.nonidealities.Discretised(args["conductance_levels"])
    G = nonideality.disturb_G(G)
    # G = crossbar.map.round_to_closest(G, args["conductance_levels"])
    utils.assert_tf_approx(G, G_exp)
    assert max_weight == max_weight_exp
