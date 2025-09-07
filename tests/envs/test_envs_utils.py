import numpy as np

from ac_solver.envs.utils import (
    convert_relators_to_presentation,
    simplify_relator_with_deque,
)


# Test convert_relators_to_presentation function
def test_convert_relators_to_presentation():
    rel1 = [1, 1, -2, -2, -2]
    rel2 = [1, 2, 1, -2, -1, -2]
    max_relator_length = 7
    arr = convert_relators_to_presentation(rel1, rel2, max_relator_length)
    expected_array = np.array(
        [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
    )
    assert np.array_equal(arr, expected_array)


def test_simplify_deque():
    rel1 = np.array([1, 1, -2, -2, -2, 0, 0], dtype=np.int8)  # This cannot be reduced
    rel2 = np.array(
        [1, 1, 2, -2, -2, 0, 0], dtype=np.int8
    )  # The middle 2s can be reduced
    rel3 = np.array(
        [1, 1, 2, -1, 1, -2, -2, 0, 0], dtype=np.int8
    )  # The middle 1's and 2's can be reduced
    rel4 = np.array(
        [2, 1, -2, -2, -2, 0, 0], dtype=np.int8
    )  # The head 2 can be reduced
    rel5 = np.array(
        [2, 1, 2, -2, -2, 0, 0], dtype=np.int8
    )  # The head and middle 2's can be reduced
    max_relator_length = 10
    arr1, _ = simplify_relator_with_deque(
        rel1, max_relator_length, cyclical=True, padded=True
    )
    arr2, _ = simplify_relator_with_deque(
        rel2, max_relator_length, cyclical=True, padded=True
    )
    arr3, _ = simplify_relator_with_deque(
        rel3, max_relator_length, cyclical=False, padded=True
    )
    arr4, _ = simplify_relator_with_deque(
        rel4, max_relator_length, cyclical=True, padded=True
    )
    arr5, _ = simplify_relator_with_deque(
        rel5, max_relator_length, cyclical=True, padded=False
    )
    arr6, _ = simplify_relator_with_deque(
        rel5, max_relator_length, cyclical=False, padded=False
    )

    expected_array1 = np.array([1, 1, -2, -2, -2, 0, 0, 0, 0, 0], dtype=np.int8)
    expected_array2 = np.array([1, 1, -2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    expected_array3 = np.array([1, 1, -2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    expected_array4 = np.array([1, -2, -2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    expected_array5 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    expected_array6 = np.array([2, 1, -2, 0, 0, 0, 0], dtype=np.int8)
    assert (
        np.array_equal(arr1, expected_array1)
        and np.array_equal(arr2, expected_array2)
        and np.array_equal(arr3, expected_array3)
        and np.array_equal(arr4, expected_array4)
        and np.array_equal(arr5, expected_array5)
        and np.array_equal(arr6, expected_array6)
    )


test_simplify_deque()
