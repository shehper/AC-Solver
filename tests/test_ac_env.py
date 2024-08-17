import pytest
import numpy as np
from rlformath.envs.ac_env import simplify_relator, is_presentation_valid, is_presentation_trivial

# Parameterized tests
@pytest.mark.parametrize(
    "relator, max_relator_length, cyclical, padded, expected_relator, expected_length",
    [
        # test case: no simplification needed
        (np.array([1, 2, 3]), 5, True, True, np.array([1, 2, 3, 0, 0]), 3),
        (np.array([1, 2, 3]), 5, True, False, np.array([1, 2, 3]), 3),
        (np.array([1, 2, 3]), 5, False, True, np.array([1, 2, 3, 0, 0]), 3),
        (np.array([1, 2, 3]), 5, False, False, np.array([1, 2, 3]), 3),

        # test case: one simplification needed
        (np.array([1, 2, -2, 3]), 5, True, True, np.array([1, 3, 0, 0, 0]), 2),
        (np.array([1, 2, -2, 3]), 5, True, False, np.array([1, 3]), 2),
        (np.array([1, 2, -2, 3]), 5, False, True, np.array([1, 3, 0, 0, 0]), 2),
        (np.array([1, 2, -2, 3]), 5, False, False, np.array([1, 3]), 2),

        # test case: two simplifications needed
        (np.array([1, 2, -2, -1]), 5, True, True, np.array([0, 0, 0, 0, 0]), 0),
        (np.array([1, 2, -2, -1]), 5, True, False, np.array([]), 0),
        (np.array([1, 2, -2, -1]), 5, False, True, np.array([0, 0, 0, 0, 0]), 0),
        (np.array([1, 2, -2, -1]), 5, False, False, np.array([]), 0),

        # test case: two simplifications needed (one after another)
        (np.array([1, -1, -2, 2]), 5, True, True, np.array([0, 0, 0, 0, 0]), 0),
        (np.array([1, -1, -2, 2]), 5, True, False, np.array([]), 0),
        (np.array([1, -1, -2, 2]), 5, False, True, np.array([0, 0, 0, 0, 0]), 0),
        (np.array([1, -1, -2, 2]), 5, False, False, np.array([]), 0),

        # test case: max_relator_length = length of simplified relator
        (np.array([1, 2, -2, 3]), 2, True, True, np.array([1, 3]), 2),
        (np.array([1, 2, -2, 3]), 2, True, False, np.array([1, 3]), 2),
        (np.array([1, 2, -2, 3]), 2, False, True, np.array([1, 3]), 2),
        (np.array([1, 2, -2, 3]), 2, False, False, np.array([1, 3]), 2),

        # test case: cyclic reduction only
        (np.array([1, 2, 3, -1]), 5, True, True, np.array([2, 3, 0, 0, 0]), 2),
        (np.array([1, 2, 3, -1]), 5, True, False, np.array([2, 3]), 2),
        (np.array([1, 2, 3, -1]), 5, False, True, np.array([1, 2, 3, -1, 0]), 4),
        (np.array([1, 2, 3, -1]), 5, False, False, np.array([1, 2, 3, -1]), 4),

        # test case: two cyclic reductions
        (np.array([1, 2, 3, -2, -1]), 6, True, True, np.array([3, 0, 0, 0, 0, 0]), 1),
        (np.array([1, 2, 3, -2, -1]), 6, True, False, np.array([3]), 1),
        (np.array([1, 2, 3, -2, -1]), 6, False, True, np.array([1, 2, 3, -2, -1, 0]), 5),
        (np.array([1, 2, 3, -2, -1]), 6, False, False, np.array([1, 2, 3, -2, -1]), 5),

        # test case: one cyclic and one ordinary reduction
        (np.array([1, 2, -2, 3, -1]), 6, True, True, np.array([3, 0, 0, 0, 0, 0]), 1),
        (np.array([1, 2, -2, 3, -1]), 6, True, False, np.array([3]), 1),
        (np.array([1, 2, -2, 3, -1]), 6, False, True, np.array([1, 3, -1, 0, 0, 0]), 3),
        (np.array([1, 2, -2, 3, -1]), 6, False, False, np.array([1, 3, -1]), 3),
    ]
)
def test_simplify_relator(relator, max_relator_length, cyclical, padded, expected_relator, expected_length):
    result = simplify_relator(relator, max_relator_length, cyclical=cyclical, padded=padded)
    assert np.array_equal(result[0], expected_relator), f"Expected {expected_relator}, but got {result[0]} \
        when relator = {relator}, max length = {max_relator_length}, cyclical = {cyclical}, padded = {padded}"
    assert result[1] == expected_length, f"Expected {expected_length}, but got {result[1]} \
        when relator = {relator}, max length = {max_relator_length}, cyclical = {cyclical}, padded = {padded}"


@pytest.mark.parametrize(
    "presentation, expected",
    [
        (np.array([1, 2, 0, 0, -2, -1, 0, 0]), True),
        (np.array([1, 0, 2, 0, -2, -1, 0, 0]), False),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0]), False),
        (np.array([1, 2, 3, 0, -3, -2, -1, 0]), True),
        (np.array([1, 0, 0, 0, 0, 0, -1, 0]), False),
        (np.array([1, 2, 0, 0, 0, -2, -1, 0]), False),
        (np.array([]), False),
        (np.array([1, 0, 0]), False),
        (np.array([1, 0, 0, 0, 0, 0, 0, 0]), False),
        (np.array([1, 2, -1, -2, 0, 0, 0, 0]), False)
    ]
)
def test_is_presentation_valid(presentation, expected):
    assert is_presentation_valid(presentation) == expected

@pytest.mark.parametrize(
    "presentation, expected",
    [
        (np.array([1, 0, 2, 0]), True),  # Trivial presentation <x, y>
        (np.array([2, 0, 1, 0]), True),  # Trivial presentation <y, x>
        (np.array([-1, 0, 2, 0]), True), # Non-trivial, contains x^-1 but invalid
        (np.array([1, 0, -2, 0]), True), # Non-trivial, contains y^-1 but invalid
        (np.array([0, 0, 0, 0]), False),  # No generators presented
        (np.array([1, 2, 0, 0]), False),  # More than one generator in a single relator
        (np.array([1, 0, 0, 0]), False),  # Incorrect relator length
        (np.array([1, 0, 0, 2]), False),  # Generators are not correctly isolated
        (np.array([-1, 0, -2, 0]), True), # All generators inverted incorrectly
        (np.array([2, 0, 1, 0]), True)   # Incorrect max_relator_length for given setup
    ]
)
def test_is_presentation_trivial(presentation, expected):
    assert is_presentation_trivial(presentation) == expected, f"test failed for presentation {presentation}; \
                                                                expected = {expected}"


# def test_simplify_assertion():
#     rel = np.array([1, -1, 2, 3, -3])
#     nrel = 2
#     try:
#         simplify_relator(rel, nrel)
#     except AssertionError as e:
#         assert str(e) == "Increase max length! Word length is bigger than maximum allowed length."
#     else:
#         assert False, "Expected an AssertionError"
