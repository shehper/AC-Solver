import numpy as np
from ac_solver.search.breadth_first import (
    bfs,
)  # Ensure this imports the bfs function correctly


def test_bfs_on_AK2():
    # Input setup: n=2 case of Akbulut-Kirby series
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    # Expected output
    expected_output = (
        True,
        # fmt: off
        [ (-1, 11), (4, 11), (11, 11), (2, 12), (4, 12), (11, 12), (9, 12), (0, 11), (5, 11), (7, 11), (3, 13), (11, 13), (9, 13), (2, 12), (8, 12), (9, 12), (3, 7), (0, 5), (0, 3), (3, 2), ],
        # fmt: on
    )

    # Call the function
    result = bfs(
        presentation=presentation, max_nodes_to_explore=int(1e6), verbose=False
    )

    # Assert the result is as expected
    assert (
        result == expected_output
    ), "The BFS function did not return the expected output"


def test_bfs_max_nodes_reached():
    # Test on AK(2) but with maximum of 10 nodes.
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])
    expected_output = (False, None)  # Assuming no solution within max nodes
    result = bfs(presentation=presentation, max_nodes_to_explore=10, verbose=False)
    assert (
        result == expected_output
    ), "BFS should stop when max_nodes_to_explore is reached"
