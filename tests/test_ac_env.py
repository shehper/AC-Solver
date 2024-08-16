import numpy as np
from rlformath.envs.ac_env import simplify_relator

def test_simplify_relator_no_inverses():
    rel = np.array([1, 2, 3])
    nrel = 5
    result = simplify_relator(rel, nrel)
    expected = (np.array([1, 2, 3, 0, 0]), 3)
    assert np.array_equal(result[0], expected[0]), f"Expected {expected[0]}, but got {result[0]}"
    assert result[1] == expected[1], f"Expected {expected[1]}, but got {result[1]}"

def test_simplify_relator_simple_inverses():
    rel = np.array([1, 2, -2, 3])
    nrel = 5
    result = simplify_relator(rel, nrel)
    expected = (np.array([1, 3, 0, 0, 0]), 2)
    assert np.array_equal(result[0], expected[0]), f"Expected {expected[0]}, but got {result[0]}"
    assert result[1] == expected[1], f"Expected {expected[1]}, but got {result[1]}"

def test_simplify_relator_full_inverses():
    rel = np.array([2, 1, -1, 3])
    nrel = 5
    result = simplify_relator(rel, nrel, cyclical=True)
    expected = (np.array([2, 3, 0, 0, 0]), 2)
    assert np.array_equal(result[0], expected[0]), f"Expected {expected[0]}, but got {result[0]}"
    assert result[1] == expected[1], f"Expected {expected[1]}, but got {result[1]}"

def test_simplify_relator_padded_false():
    rel = np.array([2, 1, -1, 3])
    nrel = 5
    result = simplify_relator(rel, nrel, cyclical=True, padded=False)
    expected = (np.array([2, 3]), 2)
    assert np.array_equal(result[0], expected[0]), f"Expected {expected[0]}, but got {result[0]}"
    assert result[1] == expected[1], f"Expected {expected[1]}, but got {result[1]}"

def test_simplify_relator_empty_relation():
    rel = np.array([1, -1, 2, -2])
    nrel = 5
    result = simplify_relator(rel, nrel)
    expected = (np.array([0, 0, 0, 0, 0]), 0)
    assert np.array_equal(result[0], expected[0]), f"Expected {expected[0]}, but got {result[0]}"
    assert result[1] == expected[1], f"Expected {expected[1]}, but got {result[1]}"

# def test_simplify_assertion():
#     rel = np.array([1, -1, 2, 3, -3])
#     nrel = 2
#     try:
#         simplify_relator(rel, nrel)
#     except AssertionError as e:
#         assert str(e) == "Increase max length! Word length is bigger than maximum allowed length."
#     else:
#         assert False, "Expected an AssertionError"

# Execute tests
test_simplify_relator_no_inverses()
test_simplify_relator_simple_inverses()
test_simplify_relator_full_inverses()
test_simplify_relator_padded_false()
test_simplify_relator_empty_relation()
# test_simplify_assertion()

print("All tests passed!")
