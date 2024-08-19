"""
Implementation of BFS for AC graph.

"""

import numpy as np
from collections import deque 
from rlformath.envs.ac_env import ACMove, is_array_valid_presentation

def bfs(presentation, 
        max_nodes_to_explore, 
        verbose=False,
        cyclically_reduce_after_moves=False):

    """
    Performs breadth-first-search on AC graph starting from the node associated to input presentation.
    Search is terminated when a node corresponding to a trivial state is found or when we have explored `max_nodes_to_explore` nodes.

    Parameters:
    presentation: A NumPy array representing a presentation.
    max_nodes_to_explore (int): The maximum number of nodes to explore during search. 
    verbose (bool): It determines whether to print information each time a presentation with smaller total length is found.
    cyclically_reduce_after_moves (bool): It determines whether 

    Returns:
    (is_search_successful, path)
    is_search_successful is a bool indicating whether a path to a trivial state is found.
    path is a list of tuples of (action, presentation_length) where action is the AC Move applied to obtain the current node
    and presentation_length is the total length of the presentation.
    """

    assert is_array_valid_presentation(presentation), f"{presentation} is not a valid presentation"

    # set initial state for search and maximum relator length allowed
    # if we encounter a presentation with a relator of length greater than max_relator_length, 
    initial_state = np.array(presentation, dtype=np.int8) # so that input may be a list or a tuple
    max_relator_length = len(presentation)//2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)
    
    # TODO: one annoying thing here is that AC moves are applied to NumPy arrays, but 
    # NumPy arrays are not hashable. So we have to go back-and-forth between converting them to
    # tuples before adding them to sets. Is there a faster soltuion?

    # add to a queue, keeping track of path length to initial state
    # a set containing states that have already been seen
    state_tup = tuple(initial_state)
    tree_nodes = {state_tup}
    init_path = [(0, total_initial_length)]
    to_explore = deque([(state_tup, init_path)]) # 
    min_length = sum(word_lengths)
    
    while (to_explore):
        state_tuple, path = to_explore.popleft()
        state = np.array(state_tuple, dtype=np.int8) # convert tuple to state
        word_lengths = [np.count_nonzero(presentation[:max_relator_length]), np.count_nonzero(presentation[max_relator_length:])]

        for action in range(1, 13):    
            new_state, new_word_lengths = ACMove(move_id=action, 
                                            presentation=state, 
                                            max_relator_length=max_relator_length, 
                                            lengths=word_lengths, 
                                            cyclical=cyclically_reduce_after_moves) 
            state_tup, new_length = tuple(new_state), sum(new_word_lengths)
            
            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f'New minimal length found: {min_length}')

            if new_length == 2:
                return True, path + [(action, new_length)]

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                to_explore.append((state_tup, path + [(action, new_length)]))

        if len(tree_nodes) >= max_nodes_to_explore:
            print(f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}")
            break

    return False, None



if __name__=='__main__':
    # AK(2)
    presentation =  np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    ans, path = bfs(presentation=presentation, 
                    max_nodes_to_explore=int(1e6))
    path(path)

    if path:
        presentation = presentation
        word_lengths = [5, 6]

        for action, _ in path[1:]:
            presentation, word_lengths = ACMove(move_id=action,
                                         presentation=presentation,
                                         max_relator_length=7,
                                         lengths=word_lengths,
                                         cyclical=False)

        print(f"Final state: {presentation}")