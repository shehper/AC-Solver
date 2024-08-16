import pytest
import numpy as np
from rlformath.envs.ac_env import simplify_relator

# Parameterized tests
@pytest.mark.parametrize(
    "relator, max_relator_length, cyclical, padded, expected_relator, expected_length",
    [
        (np.array([1, 2, 3]), 5, False, True, np.array([1, 2, 3, 0, 0]), 3),
        (np.array([1, 2, -2, 3]), 5, False, True, np.array([1, 3, 0, 0, 0]), 2),
        (np.array([2, 1, -1, 3]), 5, True, True, np.array([2, 3, 0, 0, 0]), 2),
        (np.array([2, 1, -1, 3]), 5, True, False, np.array([2, 3]), 2),
        (np.array([1, -1, 2, -2]), 5, False, True, np.array([0, 0, 0, 0, 0]), 0),
    ]
)
def test_simplify_relator(relator, max_relator_length, cyclical, padded, expected_relator, expected_length):
    result = simplify_relator(relator, max_relator_length, cyclical=cyclical, padded=padded)
    assert np.array_equal(result[0], expected_relator), f"Expected {expected_relator}, but got {result[0]}"
    assert result[1] == expected_length, f"Expected {expected_length}, but got {result[1]}"


# def test_simplify_assertion():
#     rel = np.array([1, -1, 2, 3, -3])
#     nrel = 2
#     try:
#         simplify_relator(rel, nrel)
#     except AssertionError as e:
#         assert str(e) == "Increase max length! Word length is bigger than maximum allowed length."
#     else:
#         assert False, "Expected an AssertionError"
