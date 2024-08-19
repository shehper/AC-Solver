"""
Implementation of greedy search for AC graph.

"""

import numpy as np
import heapq
from rlformath.envs.ac_env import ACMove


def greedy_search(presentation, 
                max_nodes_to_explore, 
                verbose=False,
                cyclically_reduce_after_moves=False):
    
    """
    Performs greedy search on AC graph starting from the node corresponding to input presentation.
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

    presentation = np.array(presentation, dtype=np.int8) # so that input may be a list or a tuple

    # set initial state for search and maximum relator length allowed
    # if we encounter a presentation with a relator of length greater than max_relator_length, 
    initial_state = np.array(presentation, dtype=np.int8) # so that input may be a list or a tuple
    max_relator_length = len(presentation)//2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)
    
    # add to a priority queue, keeping track of path length to initial state
    path_length = 0 
    to_explore = [(total_initial_length, path_length, tuple(initial_state), tuple(word_lengths), [(0, total_initial_length)])] 
    heapq.heapify(to_explore) 

    # a set containing states that have already been seen
    tree_nodes = set() 
    tree_nodes.add(tuple(initial_state)) 
    min_length = total_initial_length

    while (to_explore):
        _, path_length, state_tuple, word_lengths, path = heapq.heappop(to_explore)
        state = np.array(state_tuple, dtype=np.int8) # convert tuple to state
        word_lengths = list(word_lengths)

        for action in range(1, 13):    
            new_state, new_lengths = ACMove(action, state, max_relator_length, word_lengths, cyclical=cyclically_reduce_after_moves) 
            state_tup, new_length = tuple(new_state), sum(new_lengths)
            
            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f'New minimal length found: {min_length}')

            if new_length == 2:
                if verbose:
                    print(f"Found {new_state[0:1], new_state[max_relator_length:max_relator_length+1]} after exploring {len(tree_nodes)-len(to_explore)} nodes")
                    print(f"Path to a trivial state: (tuples are of form (action, length of a state)) {path + [(action, new_length)]}")
                    print(f"Total path length: {len(path)+1}")
                return True, path + [(action, new_length)]

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                heapq.heappush(to_explore, (new_length, path_length+ 1, state_tup, tuple(new_lengths), path + [(action, new_length)]))


        if len(tree_nodes) >= max_nodes_to_explore:
            print(f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}")
            break

    return False, path + [(action, new_length)]




if __name__=='__main__':
    # AK(2)
    presentation =  np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    _, path = greedy_search(presentation=presentation, max_nodes_to_explore=int(1e6), verbose=True)
    print(path)

    # check that the path actually trivializes the initial state
    if path:
        state = presentation
        lengths = [6, 5]
        max_length = 7

        for action, _ in path[1:]:
            old_state = state.copy()
            state, lengths = ACMove(action, state, max_length, lengths, cyclical=False)
        print(f"Final state: {state}")