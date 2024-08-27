"""
Implementation of greedy search for AC graph.

Example:
Trivialize Akbulut-Kirby series n=2 case "AK(2)" through greedy search as 
python greedy.py
"""

import numpy as np
import heapq
from ac_solver.envs.utils import is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove


def greedy_search(
    presentation,
    max_nodes_to_explore=10000,
    verbose=False,
    cyclically_reduce_after_moves=False,
):
    """
    Performs a greedy search on an AC graph starting from the given presentation.

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

    presentation = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple

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

    # add to a priority queue, keeping track of path length to initial state
    path_length = 0
    to_explore = [
        (
            total_initial_length,
            path_length,
            tuple(initial_state),
            tuple(word_lengths),
            [(-1, total_initial_length)],
        )
    ]
    heapq.heapify(to_explore)

    # a set containing states that have already been seen
    tree_nodes = set()
    tree_nodes.add(tuple(initial_state))
    min_length = total_initial_length

    while to_explore:
        _, path_length, state_tuple, word_lengths, path = heapq.heappop(to_explore)
        state = np.array(state_tuple, dtype=np.int8)  # convert tuple to state
        word_lengths = list(word_lengths)

        for action in range(0, 12):
            new_state, new_lengths = ACMove(
                action,
                state,
                max_relator_length,
                word_lengths,
                cyclical=cyclically_reduce_after_moves,
            )
            state_tup, new_length = tuple(new_state), sum(new_lengths)

            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f"New minimal length found: {min_length}")

            if new_length == 2:
                if verbose:
                    print(
                        f"Found {new_state[0:1], new_state[max_relator_length:max_relator_length+1]} after exploring {len(tree_nodes)-len(to_explore)} nodes"
                    )
                    print(
                        f"Path to a trivial state: (tuples are of form (action, length of a state)) {path + [(action, new_length)]}"
                    )
                    print(f"Total path length: {len(path)+1}")
                return True, path + [(action, new_length)]

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                heapq.heappush(
                    to_explore,
                    (
                        new_length,
                        path_length + 1,
                        state_tup,
                        tuple(new_lengths),
                        path + [(action, new_length)],
                    ),
                )

        if len(tree_nodes) >= max_nodes_to_explore:
            print(
                f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}"
            )
            break

    return False, path + [(action, new_length)]


if __name__ == "__main__":

    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])  # AK(2)

    ans, path = greedy_search(presentation=presentation, max_nodes_to_explore=int(1e6))

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
