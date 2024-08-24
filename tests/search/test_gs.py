import numpy as np
from ac_solver.search.greedy import (
    greedy_search,
)  # Ensure this imports the bfs function correctly


def test_gs_on_AK2():
    # Input setup: n=2 case of Akbulut-Kirby series
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    # Expected output
    expected_output = (
        True,
        [
            (0, 11),
            (12, 11),
            (5, 11),
            (3, 12),
            (12, 12),
            (5, 12),
            (10, 12),
            (1, 11),
            (2, 13),
            (9, 13),
            (7, 13),
            (10, 13),
            (1, 12),
            (2, 11),
            (7, 11),
            (9, 11),
            (4, 8),
            (7, 8),
            (3, 5),
            (6, 5),
            (2, 3),
            (9, 3),
            (3, 2),
        ],
    )

    # Call the function
    result = greedy_search(
        presentation=presentation, max_nodes_to_explore=int(1e6), verbose=False
    )

    # Assert the result is as expected
    assert (
        result == expected_output
    ), "greedy search function did not return the expected output"
