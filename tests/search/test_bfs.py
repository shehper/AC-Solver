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
        [
            (0, 11),
            (5, 11),
            (12, 11),
            (3, 12),
            (5, 12),
            (12, 12),
            (10, 12),
            (1, 11),
            (6, 11),
            (8, 11),
            (4, 13),
            (12, 13),
            (10, 13),
            (3, 12),
            (9, 12),
            (10, 12),
            (4, 7),
            (1, 5),
            (1, 3),
            (4, 2),
        ],
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
