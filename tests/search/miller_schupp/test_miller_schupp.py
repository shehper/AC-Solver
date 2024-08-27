import pytest
import numpy as np
from ac_solver.search.miller_schupp.miller_schupp import (
    trivialize_miller_schupp_through_search,
    generate_miller_schupp_presentations,
)
from ac_solver.search.greedy import greedy_search
from ac_solver.search.breadth_first import bfs

# TODO: write more tests involving more ranges of n, w_len and search_fn.


@pytest.mark.parametrize(
    "n, max_w_len", [(n, max_w_len) for n in range(1, 4) for max_w_len in range(1, 4)]
)
def test_generate_miller_schupp_presentations(n, max_w_len):
    ms_presentations = generate_miller_schupp_presentations(n=n, max_w_len=max_w_len)
    assert isinstance(
        ms_presentations, dict
    ), f"expected ms_presentations to be a dict; got {type(ms_presentations)}"
    assert list(ms_presentations.keys()) == list(
        range(1, max_w_len + 1)
    ), f"expect output dict to have a key for each value of w_len in range [1, max_w_len]; got {ms_presentations.keys()}"

    for w_len in range(1, max_w_len + 1):
        for presentation in ms_presentations[w_len]:
            assert (
                len(presentation) % 2 == 0
            ), "expect length of each presentation to be even as number of relators is 2."

            max_relator_length = len(presentation) // 2
            relator1 = presentation[:max_relator_length]
            relator2 = presentation[max_relator_length:]
            assert (
                np.count_nonzero(relator1) == 2 * n + 3
            ), f"expect relator 1 to have length = {2 * n + 3}; got {np.count_nonzero(relator1)}"
            assert (
                relator2[0] == -1
            ), f"expect the first letter in relator 2 to be -1; got {relator2[0]}"


def test_number_of_miller_schupp_presentations():
    ms_presentations = generate_miller_schupp_presentations(n=1, max_w_len=7)
    total_number_of_presentations = sum([len(x) for x in ms_presentations.values()])
    assert (
        len(ms_presentations[1]) == 2
    ), f"expect 2 presentations for w_len = 1; got {len(ms_presentations[1])}"
    assert (
        len(ms_presentations[2]) == 2
    ), f"expect 2 presentations for w_len = 2; got {len(ms_presentations[2])}"
    assert (
        len(ms_presentations[3]) == 2
    ), f"expect 3 presentations for w_len = 2; got {len(ms_presentations[3])}"
    assert (
        total_number_of_presentations == 170
    ), f"expect got {total_number_of_presentations}"


def test_trivialize_miller_schupp_through_greedy_search():

    # Case 1
    solved_rels, unsolved_rels, solved_paths = trivialize_miller_schupp_through_search(
        min_n=1,
        max_n=2,
        min_w_len=1,
        max_w_len=2,
        max_nodes_to_explore=int(1e6),
        search_fn=greedy_search,
    )

    expected_solved_rels = [
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # fmt: off
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # fmt: on
    ]

    expected_unsolved_rels = []

    expected_solved_paths = [
        [(-1, 7), (5, 7), (8, 7), (3, 5), (9, 3), (2, 2)],
        [(-1, 7), (1, 7), (7, 5), (5, 3), (3, 3), (2, 2)],
        [(-1, 8), (8, 8), (3, 5), (6, 5), (2, 3), (1, 2)],
        [(-1, 8), (1, 7), (7, 5), (3, 4), (2, 3), (2, 2)],
        [(-1, 9), (7, 9), (5, 9), (1, 7), (5, 5), (11, 5), (3, 3), (0, 2)],
        [(-1, 9), (1, 9), (7, 7), (5, 5), (5, 3), (3, 3), (2, 2)],
        # fmt: off
        [ (-1, 10), (8, 10), (3, 9), (11, 9), (6, 9), (3, 8), (6, 8), (1, 7), (7, 5), (2, 5), (8, 3), (1, 2), ],
        # fmt: on
        [(-1, 10), (1, 9), (7, 7), (5, 5), (3, 4), (2, 3), (2, 2)],
    ]

    assert solved_rels == expected_solved_rels
    assert unsolved_rels == expected_unsolved_rels
    assert solved_paths == expected_solved_paths

    # Case 2
    solved_rels, unsolved_rels, solved_paths = trivialize_miller_schupp_through_search(
        min_n=3,
        max_n=4,
        min_w_len=3,
        max_w_len=4,
        max_nodes_to_explore=int(1e4),
        search_fn=greedy_search,
    )

    expected_solved_rels = [
        # fmt: off
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # fmt: on
    ]

    expected_unsolved_rels = [
        # fmt: off
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # fmt: on
    ]

    expected_solved_paths = [
        # fmt: off
        [ (-1, 13), (8, 13), (3, 11), (11, 11), (5, 11), (4, 11), (7, 11), (1, 7), (11, 5), (0, 4), (0, 3), (0, 2), ],
        [(-1, 13), (1, 11), (7, 9), (0, 6), (5, 4), (7, 4), (3, 3), (3, 2)],
        [ (-1, 14), (5, 14), (5, 14), (5, 14), (4, 14), (1, 13), (9, 13), (1, 14), (7, 14), (7, 14), (1, 13), (6, 13), (7, 13), (1, 12), (11, 12), (1, 13), (9, 13), (7, 13), (10, 13), (3, 12), (5, 12), (7, 12), (1, 13), (7, 13), (7, 13), (1, 12), (7, 12), (3, 11), (11, 11), (11, 11), (3, 10), (11, 10), (11, 10), (3, 9), (5, 9), (7, 9), (3, 8), (2, 7), (5, 7), (3, 5), (2, 4), (8, 4), (2, 3), (8, 3), (2, 2), ],
        [(-1, 14), (8, 14), (3, 9), (4, 9), (2, 7), (8, 5), (1, 4), (1, 3), (1, 2)],
        [ (-1, 14), (7, 14), (5, 14), (8, 14), (8, 14), (3, 13), (5, 13), (3, 14), (11, 14), (11, 14), (3, 13), (11, 13), (11, 13), (5, 13), (5, 13), (4, 13), (3, 12), (8, 12), (11, 12), (3, 13), (7, 13), (10, 13), (9, 13), (1, 12), (11, 12), (1, 13), (7, 13), (9, 13), (1, 12), (11, 12), (5, 12), (5, 12), (6, 12), (4, 12), (3, 11), (8, 11), (11, 11), (3, 12), (11, 12), (11, 12), (3, 11), (10, 11), (11, 11), (11, 11), (1, 10), (11, 10), (2, 9), (7, 9), (3, 5), (0, 4), (0, 3), (4, 3), (0, 2), ],
        [(-1, 14), (1, 11), (7, 9), (3, 6), (2, 5), (2, 4), (2, 3), (2, 2)],
        [ (-1, 15), (8, 15), (3, 13), (11, 13), (6, 13), (3, 11), (6, 11), (6, 11), (1, 9), (7, 7), (2, 6), (8, 4), (1, 3), (1, 2), ],
        [(-1, 15), (1, 13), (7, 11), (5, 9), (0, 6), (5, 4), (7, 4), (3, 3), (3, 2)],
        [ (-1, 16), (8, 16), (3, 13), (11, 13), (5, 13), (4, 13), (7, 13), (1, 8), (11, 6), (0, 5), (0, 4), (0, 3), (0, 2), ],
        [(-1, 16), (1, 13), (7, 11), (0, 7), (5, 5), (7, 5), (3, 4), (3, 3), (3, 2)],
        # fmt: on
    ]

    assert solved_rels == expected_solved_rels
    assert unsolved_rels == expected_unsolved_rels
    assert solved_paths == expected_solved_paths


def test_trivialize_miller_schupp_through_bfs():

    solved_rels, unsolved_rels, solved_paths = trivialize_miller_schupp_through_search(
        min_n=1,
        max_n=2,
        min_w_len=1,
        max_w_len=2,
        max_nodes_to_explore=int(1e4),
        search_fn=bfs,
    )

    expected_solved_rels = [
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    expected_unsolved_rels = [
        # fmt: off
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [ -1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # fmt: on
    ]

    expected_solved_paths = [[(-1, 7), (1, 7), (7, 5), (0, 4), (5, 2)]]

    assert solved_rels == expected_solved_rels
    assert unsolved_rels == expected_unsolved_rels
    assert solved_paths == expected_solved_paths
