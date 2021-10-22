import numpy as np
import pytest
import mingann


w_to_G_testdata = [
        (
            {
                "weights": np.array([
                    [4.0, 2.0, -5.0],
                    [-1.0, 0.0, 1.0],
                    ]),
                "conductance_levels": np.array(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                    ),
                },
            [
                np.array([
                    [5.0, 1.0, 3.0, 1.0, 1.0, 6.0],
                    [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
                    ]),
                5.0
                ],
            ),
        (
            {
                "weights": np.array([
                    [4.2, 2.1, -5.0],
                    [-1.0, 0.9, 0.2],
                    ]),
                "conductance_levels": np.array(
                    [1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5, 5.0e-5, 6.0e-5]
                    ),
                },
            [
                np.array([
                    [5.0e-5, 1.0e-5, 3.0e-5, 1.0e-5, 1.0e-5, 6.0e-5],
                    [1.0e-5, 2.0e-5, 2.0e-5, 1.0e-5, 1.0e-5, 1.0e-5],
                    ]),
                5.0
                ],
            ),
        (
            {
                "weights": np.array([
                    [1.0, 2.0, 3.0],
                    [-1.0, -2.0, 3.0],
                    ]),
                "conductance_levels": np.array(
                    [1.1, 2.0, 2.1]
                    ),
                },
            [
                np.array([
                    [1.1, 1.1, 2.0, 1.1, 2.1, 1.1],
                    [1.1, 1.1, 1.1, 2.0, 2.1, 1.1],
                    ]),
                3.0
                ],
            ),
        ]


@pytest.mark.parametrize("args,expected", w_to_G_testdata)
def test_w_to_G(args, expected):
    G_exp, max_weight_exp = expected
    G, max_weight = mingann.map.w_to_G(**args)
    np.testing.assert_almost_equal(G, G_exp)
    np.testing.assert_almost_equal(max_weight, max_weight_exp)
