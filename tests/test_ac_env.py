import pytest
import numpy as np
from rlformath.envs.ac_env import simplify_relator, is_array_valid_presentation, \
                                  is_presentation_trivial, generate_trivial_states, \
                                  simplify_presentation, concatenate_relators, conjugate

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
def test_is_array_valid_presentation(presentation, expected):
    assert is_array_valid_presentation(presentation) == expected

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


@pytest.mark.parametrize("max_relator_length", [1, 2, 3, 4])
def test_generate_trivial_states_structure(max_relator_length):
    states = generate_trivial_states(max_relator_length)
    assert states.shape == (8, 2 * max_relator_length)
    for state in states:
        # Check if the first and max_relator_length + 1 entries are the only non-zero values
        assert np.count_nonzero(state) == 2
        # Check if all zeros are correctly placed
        assert np.count_nonzero(state[1:max_relator_length]) == 0
        assert np.count_nonzero(state[max_relator_length + 1:]) == 0
        assert abs(state[0]) != abs(state[max_relator_length])

@pytest.mark.parametrize(
    "presentation, max_relator_length, lengths_of_words, expected_output, expected_lengths",
    [
        (np.array([1, 0, 2, 0]), 2, [1, 1], np.array([1, 0, 2, 0]), [1, 1]),  # Simple case
        (np.array([1, 2, -2, -2, -1, 1]), 3, [3, 3], np.array([1, 0, 0, -2, 0, 0]), [1, 1]),  # Proper simplification
        (np.array([1, 2, -1, 0, 2, -2, -1, 0]), 4, [3, 3], np.array([2, 0, 0, 0, -1, 0, 0, 0]), [1, 1]),  # Full utilization of relator length
    ]
)
def test_simplify_presentation(presentation, max_relator_length, lengths_of_words, expected_output, expected_lengths):
    simplified_presentation, simplified_lengths = simplify_presentation(presentation, max_relator_length, lengths_of_words)
    assert np.array_equal(simplified_presentation, expected_output), "Simplified presentation does not match expected"
    assert simplified_lengths == expected_lengths, "Simplified lengths do not match expected"

@pytest.mark.parametrize(
    "rels, nrel, i, j, sign, lengths, expected_rels, expected_lengths",
    [
        # test case: no change as max_relator_length < length of concatenated relator
        (np.array([1, 2, 0, 3, 4, 0]), 3, 0, 1, 1, [2, 2], np.array([1, 2, 0, 3, 4, 0]), [2, 2]),

        # test case: simple concatenation
        (np.array([1, 2, 0, 0, 3, 4, 0, 0]), 4, 0, 1, 1, [2, 2], np.array([1, 2, 3, 4, 3, 4, 0, 0]), [4, 2]),
        (np.array([1, 2, 0, 0, 3, 4, 0, 0]), 4, 0, 1, -1, [2, 2], np.array([1, 2, -4, -3, 3, 4, 0, 0]), [4, 2]),
        (np.array([1, 2, 0, 0, 3, 4, 0, 0]), 4, 1, 0, 1, [2, 2], np.array([1, 2, 0, 0, 3, 4, 1, 2]), [2, 4]),
        (np.array([1, 2, 0, 0, 3, 4, 0, 0]), 4, 1, 0, -1, [2, 2], np.array([1, 2, 0, 0, 3, 4, -2, -1]), [2, 4]),

        # test case: concatenation with simplify
        (np.array([1, 2, 0, 0, -2, 1, 0, 0]), 4, 0, 1, 1, [2, 2], np.array([1, 1, 0, 0, -2, 1, 0, 0]), [2, 2]),
        (np.array([1, 2, 0, 0, 1, -2, -1, 0]), 4, 1, 0, 1, [2, 3], np.array([1, 2, 0, 0, 1, 0, 0, 0]), [2, 1]),
        # the following two cases are interesting.. because they would give different output if we performed
        # cyclical reduction, which we do not.
        (np.array([1, 1, 0, 0, 1, -2, 0, 0]), 4, 0, 1, -1, [2, 2], np.array([1, 1, 2, -1, 1, -2, 0, 0]), [4, 2]),
        (np.array([1, 2, 0, 0, 1, -2, -1, 0]), 4, 1, 0, -1, [2, 3], np.array([1, 2, 0, 0, 1, -2, -1, 0]), [2, 3]),
        
        # Boundary conditions: first relator is shorter
        (np.array([1, 0, 0, 1, 2, 3]), 3, 0, 1, 1, [1, 3], np.array([1, 0, 0, 1, 2, 3]), [1, 3]),
        # No space for simplification: new size exceeds nrel
        (np.array([1, 2, 3, 4, 5, 6]), 3, 0, 1, 1, [3, 3], np.array([1, 2, 3, 4, 5, 6]), [3, 3]),
        # Complex case with partial cancellation and inversion
        (np.array([1, -2, 0, 2, -1, 0]), 3, 0, 1, -1, [2, 2], np.array([1, -2, 0, 2, -1, 0]), [2, 2])
    ]
)
def test_concatenate_relators(rels, nrel, i, j, sign, lengths, expected_rels, expected_lengths):
    result_rels, result_lengths = concatenate_relators(rels, nrel, i, j, sign, lengths)
    assert np.array_equal(result_rels, expected_rels), f"Relators do not match expected results \
                                                        when rels = {rels}, expected_rels = {expected_rels}, \
                                                            got result = {result_rels}"
    assert result_lengths == expected_lengths, "Lengths do not match expected results"

@pytest.mark.parametrize(
    "rels, nrel, i, j, sign, lengths, expected_rels, expected_lengths",
    [
        # Basic conjugation without cancellation
        (np.array([1, 2, 0, 2, 0, 0]), 3, 0, 2, 1, [2, 1], np.array([2, 1, 0, 2, 0, 0]), [2, 1]),
        (np.array([2, 0, 0, 1, 2, 0]), 3, 1, 2, 1, [1, 2], np.array([2, 0, 0, 2, 1, 0]), [1, 2]),

        # Basic conjugation without cancellation
        (np.array([1, 0, 0, 2, 0, 0]), 3, 0, 2, 1, [1, 1], np.array([2, 1, -2, 2, 0, 0]), [3, 1]),
        (np.array([1, -2, 0, 0, 1, 0, 0, 0]), 4, 0, 2, 1, [2, 1], np.array([2, 1, -2, -2, 1, 0, 0, 0]), [4, 1]),
        (np.array([1, 0, 0, 0, 1, -2, 0, 0]), 4, 1, 2, 1, [1, 2], np.array([1, 0, 0, 0, 2, 1, -2, -2]), [1, 4]),
        
        # Conjugation with cancellation at the end
        (np.array([2, 1, 0, 1, 0, 0]), 3, 0, 2, -1, [2, 1], np.array([1, 2, 0, 1, 0, 0]), [2, 1]),
        (np.array([2, 0, 0, 0, 1, 2, 0, 0]), 4, 1, 2, 1, [1, 2], np.array([2, 0, 0, 0, 2, 1, 0, 0]), [1, 2]),

        # Conjugation with cancellation at the start
        (np.array([2, 1, 0, 1, 0, 0]), 3, 0, 2, -1, [2, 1], np.array([1, 2, 0, 1, 0, 0]), [2, 1]),
        (np.array([2, 0, 0, 0, 1, 2, 0, 0]), 4, 1, 1, -1, [1, 2], np.array([2, 0, 0, 0, 2, 1, 0, 0]), [1, 2]),        

        # Conjugation with cancellation at both ends
        (np.array([1, 2, -1, 2, 0, 0]), 3, 0, 1, -1, [3, 1], np.array([2, 0, 0, 2, 0, 0]), [1, 1]),
        
        (np.array([2, 0, 0, 0, 1, 2, 0, 0]), 4, 0, 2, 1, [1, 2], np.array([2, 0, 0, 0, 1, 2, 0, 0]), [1, 2]),
        (np.array([1, 0, 0, 0, 1, 2, -1, 0]), 4, 1, 1, -1, [1, 3], np.array([1, 0, 0, 0, 2, 0, 0, 0]), [1, 1]),

        # No change due to insufficient max length
        (np.array([1, 2, -1, 2, 0, 0]), 3, 0, 2, 1, [3, 1], np.array([1, 2, -1, 2, 0, 0]), [3, 1]),
        
    ]
)
def test_conjugate(rels, nrel, i, j, sign, lengths, expected_rels, expected_lengths):
    result_rels, result_lengths = conjugate(rels, nrel, i, j, sign, lengths)
    assert np.array_equal(result_rels, expected_rels), "Resulting relators do not match expected results"
    assert result_lengths == expected_lengths, "Resulting lengths do not match expected results"

# def test_simplify_assertion():
#     rel = np.array([1, -1, 2, 3, -3])
#     nrel = 2
#     try:
#         simplify_relator(rel, nrel)
#     except AssertionError as e:
#         assert str(e) == "Increase max length! Word length is bigger than maximum allowed length."
#     else:
#         assert False, "Expected an AssertionError"
