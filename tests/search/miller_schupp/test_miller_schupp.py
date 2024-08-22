import pytest 
import numpy as np
from rlformath.search.miller_schupp.miller_schupp import trivialize_miller_schupp_through_search, generate_miller_schupp_presentations
from rlformath.search.greedy import greedy_search

# TODO: write more tests involving more ranges of n, w_len and search_fn.

@pytest.mark.parametrize("n, max_w_len", [(n, max_w_len) for n in range(1, 4) for max_w_len in range(1, 4)])
def test_generate_miller_schupp_presentations(n, max_w_len):
    ms_presentations = generate_miller_schupp_presentations(n=n, max_w_len=max_w_len)
    assert isinstance(ms_presentations, dict), f"expected ms_presentations to be a dict; got {type(ms_presentations)}"
    assert list(ms_presentations.keys()) == list(range(1, max_w_len + 1)), f"expect output dict to have a key for each value of w_len in range [1, max_w_len]; got {ms_presentations.keys()}"

    for w_len in range(1, max_w_len + 1):
        for presentation in ms_presentations[w_len]:
            max_relator_length = len(presentation) // 2
            relator1 = presentation[:max_relator_length]
            relator2 = presentation[max_relator_length:]
            assert np.count_nonzero(relator1) == 2 * n + 3, f"expect relator 1 to have length = {2 * n + 3}; got {np.count_nonzero(relator1)}"
            assert relator2[0] == -1, f"expect the first letter in relator 2 to be -1; got {relator2[0]}"


def test_number_of_miller_schupp_presentations():
    ms_presentations = generate_miller_schupp_presentations(n=1, max_w_len=7)
    total_number_of_presentations = sum([len(x) for x in ms_presentations.values()])
    assert len(ms_presentations[1]) == 2, f"expect 2 presentations for w_len = 1; got {len(ms_presentations[1])}"
    assert len(ms_presentations[2]) == 2, f"expect 2 presentations for w_len = 2; got {len(ms_presentations[2])}"
    assert len(ms_presentations[3]) == 2, f"expect 3 presentations for w_len = 2; got {len(ms_presentations[3])}"
    assert total_number_of_presentations == 170, f"expect got {total_number_of_presentations}"


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

    expected_solved_rels = [[-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    expected_unsolved_rels = []

    expected_solved_paths = [[(0, 7), (6, 7), (9, 7), (4, 5), (10, 3), (3, 2)], 
                             [(0, 7), (2, 7), (8, 5), (6, 3), (4, 3), (3, 2)], 
                             [(0, 8), (9, 8), (4, 5), (7, 5), (3, 3), (2, 2)], 
                             [(0, 8), (2, 7), (8, 5), (4, 4), (3, 3), (3, 2)], 
                             [(0, 9), (8, 9), (6, 9), (2, 7), (6, 5), (12, 5), (4, 3), (1, 2)], 
                             [(0, 9), (2, 9), (8, 7), (6, 5), (6, 3), (4, 3), (3, 2)], 
                             [(0, 10), (9, 10), (4, 9), (12, 9), (7, 9), (4, 8), (7, 8), (2, 7), (8, 5), (3, 5), (9, 3), (2, 2)], 
                             [(0, 10), (2, 9), (8, 7), (6, 5), (4, 4), (3, 3), (3, 2)]]

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

    expected_solved_rels = [[-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    expected_unsolved_rels = [[-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                              [-1, 2, 2, 2, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                              [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                              [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                              [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                              [-1, 2, 2, 2, 2, 1, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -2, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    expected_solved_paths = [[(0, 13), (9, 13), (4, 11), (12, 11), (6, 11), (5, 11), (8, 11), (2, 7), (12, 5), (1, 4), (1, 3), (1, 2)], 
                             [(0, 13), (2, 11), (8, 9), (1, 6), (6, 4), (8, 4), (4, 3), (4, 2)], 
                             [(0, 14), (6, 14), (6, 14), (6, 14), (5, 14), (2, 13), (10, 13), (2, 14), (8, 14), (8, 14), (2, 13), (7, 13), (8, 13), 
                              (2, 12), (12, 12), (2, 13), (10, 13), (8, 13), (11, 13), (4, 12), (6, 12), (8, 12), (2, 13), (8, 13), (8, 13), (2, 12), 
                              (8, 12), (4, 11), (12, 11), (12, 11), (4, 10), (12, 10), (12, 10), (4, 9), (6, 9), (8, 9), (4, 8), (3, 7), (6, 7), 
                              (4, 5), (3, 4), (9, 4), (3, 3), (9, 3), (3, 2)], 
                              [(0, 14), (9, 14), (4, 9), (5, 9), (3, 7), (9, 5), (2, 4), (2, 3), (2, 2)], 
                              [(0, 14), (8, 14), (6, 14), (9, 14), (9, 14), (4, 13), (6, 13), (4, 14), (12, 14), (12, 14), (4, 13), (12, 13), (12, 13), 
                               (6, 13), (6, 13), (5, 13), (4, 12), (9, 12), (12, 12), (4, 13), (8, 13), (11, 13), (10, 13), (2, 12), (12, 12), (2, 13), 
                               (8, 13), (10, 13), (2, 12), (12, 12), (6, 12), (6, 12), (7, 12), (5, 12), (4, 11), (9, 11), (12, 11), (4, 12), (12, 12), 
                               (12, 12), (4, 11), (11, 11), (12, 11), (12, 11), (2, 10), (12, 10), (3, 9), (8, 9), (4, 5), (1, 4), (1, 3), (5, 3), (1, 2)], 
                               [(0, 14), (2, 11), (8, 9), (4, 6), (3, 5), (3, 4), (3, 3), (3, 2)], 
                               [(0, 15), (9, 15), (4, 13), (12, 13), (7, 13), (4, 11), (7, 11), (7, 11), (2, 9), (8, 7), (3, 6), (9, 4), (2, 3), (2, 2)], 
                               [(0, 15), (2, 13), (8, 11), (6, 9), (1, 6), (6, 4), (8, 4), (4, 3), (4, 2)], 
                               [(0, 16), (9, 16), (4, 13), (12, 13), (6, 13), (5, 13), (8, 13), (2, 8), (12, 6), (1, 5), (1, 4), (1, 3), (1, 2)], 
                               [(0, 16), (2, 13), (8, 11), (1, 7), (6, 5), (8, 5), (4, 4), (4, 3), (4, 2)]]

    assert solved_rels == expected_solved_rels
    assert unsolved_rels == expected_unsolved_rels
    assert solved_paths == expected_solved_paths


# def test_trivialize_miller_schupp_through_bfs():

#     solved_rels, unsolved_rels, solved_paths = trivialize_miller_schupp_through_search(
#         min_n=1,
#         max_n=2,
#         min_w_len=1,
#         max_w_len=2,
#         search_fn=bfs,
#     )

#     expected_solved_rels = [[-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                             [-1, 2, 2, 1, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#     expected_unsolved_rels = []

#     expected_solved_paths = [[(0, 7), (6, 7), (9, 7), (4, 5), (10, 3), (3, 2)], 
#                              [(0, 7), (2, 7), (8, 5), (6, 3), (4, 3), (3, 2)], 
#                              [(0, 8), (9, 8), (4, 5), (7, 5), (3, 3), (2, 2)], 
#                              [(0, 8), (2, 7), (8, 5), (4, 4), (3, 3), (3, 2)], 
#                              [(0, 9), (8, 9), (6, 9), (2, 7), (6, 5), (12, 5), (4, 3), (1, 2)], 
#                              [(0, 9), (2, 9), (8, 7), (6, 5), (6, 3), (4, 3), (3, 2)], 
#                              [(0, 10), (9, 10), (4, 9), (12, 9), (7, 9), (4, 8), (7, 8), (2, 7), (8, 5), (3, 5), (9, 3), (2, 2)], 
#                              [(0, 10), (2, 9), (8, 7), (6, 5), (4, 4), (3, 3), (3, 2)]]

#     assert solved_rels == expected_solved_rels
#     assert unsolved_rels == expected_unsolved_rels
#     assert solved_paths == expected_solved_paths