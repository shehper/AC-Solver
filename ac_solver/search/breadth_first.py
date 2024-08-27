"""
Implementation of BFS for AC graph.

Example:
Trivialize Akbulut-Kirby series n=2 case "AK(2)" through BFS as 
python breadth_first.py
"""

import numpy as np
from collections import deque
from ac_solver.envs.utils import is_array_valid_presentation, is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove


def bfs(
    presentation,
    max_nodes_to_explore=10000,
    verbose=False,
    cyclically_reduce_after_moves=False,
):
    """
    Performs a breadth-first search on an AC graph starting from the given presentation.

    Parameters:
        presentation (np.ndarray): Initial presentation as a NumPy array.
        max_nodes_to_explore (int, optional): Max nodes to explore before termination (default: 10000).
        verbose (bool, optional): Print updates when shorter presentations are found (default: False).
        cyclically_reduce_after_moves (bool, optional): Apply cyclic reduction after each move (default: False).

    Returns:
        tuple: (is_search_successful, path)
            - is_search_successful (bool): Whether a trivial state was found.
            - path (list of tuple): Sequence of (action, presentation_length).
    """

    assert is_array_valid_presentation(
        presentation
    ), f"{presentation} is not a valid presentation"

    # set initial state for search and maximum relator length allowed
    # if we encounter a presentation with a relator of length greater than max_relator_length,
    initial_state = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple
    max_relator_length = len(presentation) // 2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)

    # add to a queue, keeping track of path length to initial state
    # a set containing states that have already been seen
    state_tup = tuple(initial_state)
    tree_nodes = {state_tup}
    init_path = [(-1, total_initial_length)]
    to_explore = deque([(state_tup, init_path)])  #
    min_length = sum(word_lengths)

    while to_explore:
        state_tuple, path = to_explore.popleft()
        state = np.array(state_tuple, dtype=np.int8)  # convert tuple to state
        word_lengths = [
            np.count_nonzero(presentation[:max_relator_length]),
            np.count_nonzero(presentation[max_relator_length:]),
        ]

        for action in range(0, 12):
            new_state, new_word_lengths = ACMove(
                move_id=action,
                presentation=state,
                max_relator_length=max_relator_length,
                lengths=word_lengths,
                cyclical=cyclically_reduce_after_moves,
            )
            state_tup, new_length = tuple(new_state), sum(new_word_lengths)

            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f"New minimal length found: {min_length}")

            if new_length == 2:
                return True, path + [(action, new_length)]

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                to_explore.append((state_tup, path + [(action, new_length)]))

        if len(tree_nodes) >= max_nodes_to_explore:
            print(
                f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}"
            )
            break

    return False, None


if __name__ == "__main__":

    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])  # AK(2)

    ans, path = bfs(presentation=presentation, max_nodes_to_explore=int(1e6))

    if path:
        print(
            f"""
              Presentation {presentation} solved!
              Path length: {len(path)}
              """
        )
        print("Checking whether this path actually leads to a trivial state..")
        word_lengths = [5, 6]

        for action, _ in path[1:]:
            presentation, word_lengths = ACMove(
                move_id=action,
                presentation=presentation,
                max_relator_length=7,
                lengths=word_lengths,
                cyclical=False,
            )

        print(f"Final state achieved: {presentation}")
        print(f"Is trivial? {is_presentation_trivial(presentation)}")
