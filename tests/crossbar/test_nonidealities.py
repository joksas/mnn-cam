import pytest
import tensorflow as tf

from mnn import crossbar
from tests import utils

# infinity
inf = float("inf")

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


line_resistance_mapping_testdata = [
    (
        {
            "R": tf.constant(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                ],
                dtype=tf.float32,
            ),
            "num_word_lines": 2,
            "num_bit_lines": 4,
        },
        tf.constant(
            [
                [
                    [[1, 2, 3, 4], [5, 6, 7, 8]],
                    ],
                [
                    [[9, 10, 11, 12], [13, 14, 15, 16]],
                ],
                [
                    [[17, 18, 19, 20], [inf, inf, inf, inf]],
                    ]
            ],
            dtype=tf.float32,
        ),
    ),
]

@pytest.mark.parametrize("args,expected", line_resistance_mapping_testdata)
def test_map_resistances_to_crossbars(args, expected):
    R_exp = expected
    R = crossbar.nonidealities.map_resistances_to_crossbars(**args)
    utils.assert_tf_approx(R, R_exp)

line_resistance_testdata = [
    (
        {
            "R": tf.constant(
                [
                    [1/1, 1/2, 1/3, 1/4],
                    [1/5, 1/6, 1/7, 1/8],
                    [1/9, 1/10, 1/11, 1/12],
                    [1/13, 1/14, 1/15, 1/16],
                    [1/17, 1/18, 1/19, 1/20],
                ],
                dtype=tf.float32,
            ),
            "V": tf.constant(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                    [13, 14, 15],
                ],
                dtype=tf.float32,
            ),
            "num_word_lines": 2,
            "num_bit_lines": 4,
            "word_line_r": 0,
            "bit_line_r": 0,
        },
        # Output currents:
        tf.constant(
            [
                [
                    1*1 + 4*5 + 7*9 + 10*13 + 13*17,
                    1*2 + 4*6 + 7*10 + 10*14 + 13*18,
                    1*3 + 4*7 + 7*11 + 10*15 + 13*19,
                    1*4 + 4*8 + 7*12 + 10*16 + 13*20,
                    ],
                [
                    2*1 + 5*5 + 8*9 + 11*13 + 14*17,
                    2*2 + 5*6 + 8*10 + 11*14 + 14*18,
                    2*3 + 5*7 + 8*11 + 11*15 + 14*19,
                    2*4 + 5*8 + 8*12 + 11*16 + 14*20,
                    ],
                [
                    3*1 + 6*5 + 9*9 + 12*13 + 15*17,
                    3*2 + 6*6 + 9*10 + 12*14 + 15*18,
                    3*3 + 6*7 + 9*11 + 12*15 + 15*19,
                    3*4 + 6*8 + 9*12 + 12*16 + 15*20,
                    ],
            ],
            dtype=tf.float32,
        ),
    ),
]

@pytest.mark.parametrize("args,expected", line_resistance_testdata)
def test_line_resistance(args, expected):
    R_exp = expected
    output_currents = crossbar.nonidealities.line_resistance(**args)
    utils.assert_tf_approx(output_currents, R_exp)
