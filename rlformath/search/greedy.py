"""
Implementation of greedy search for AC graph.

"""

import numpy as np
import heapq
from rlformath.envs.ac_env import ACMove

len_words = np.count_nonzero

def gs(presentation=np.array([]), rel1=None, rel2=None, max_length=None, 
        cap_num_nodes=True, max_nodes_to_explore=int(2e5), verbose=True,
        full_simplify_moves=False):
    # TODO: This is hard to work with. Let rels be either a numpy array or a tuple of
    # (rel1, rel2, max_length).
    """Takes either a numpy array/list 'rels' of shape (2*max_length,) or 
    takes rel1, rel2, max_length and converts them into this shape."""
    presentation = np.array(presentation, dtype=np.int8) # so that input may be a list or a tuple
    if presentation.any():
        initial_state = presentation
        max_length = len(presentation)//2
        lengths = [len_words(presentation[:max_length]), len_words(presentation[max_length:])]
    else:
        assert rel1 and rel2 and max_length, "Either give presentation or all rel1, rel2, and max_length"
        initial_state = to_array(rel1, rel2, max_length) 
        lengths = [len(rel1), len(rel2)]
    
    # add to a priority queue, keeping track of path length to initial state
    path_length = 0 
    to_explore = [(sum(lengths), path_length, tuple(initial_state), tuple(lengths), [(0, sum(lengths))])] 
    heapq.heapify(to_explore) 

    # a set containing states that have already been seen
    tree_nodes = set() 
    tree_nodes.add(tuple(initial_state)) 
    min_length = sum(lengths)

    while (to_explore):
        _, path_length, state_tuple, lengths, path = heapq.heappop(to_explore)
        state = np.array(state_tuple, dtype=np.int8) # convert tuple to state
        lengths = list(lengths)

        for action in range(1, 13):    
            new_state, new_lengths = ACMove(action, state, max_length, lengths, cyclical=full_simplify_moves) 
            state_tup, new_length = tuple(new_state), sum(new_lengths)
            
            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f'New minimal length found: {min_length}')

            if new_length == 2:
                if verbose:
                    print(f"Found {new_state[0:1], new_state[max_length:max_length+1]} after exploring {len(tree_nodes)-len(to_explore)} nodes")
                    print(f"Path to a trivial state: (tuples are of form (action, length of a state)) {path + [(action, new_length)]}")
                    print(f"Total path length: {len(path)+1}")
                return True, path + [(action, new_length)]

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                heapq.heappush(to_explore, (new_length, path_length+ 1, state_tup, tuple(new_lengths), path + [(action, new_length)]))


        if cap_num_nodes and len(tree_nodes) >= max_nodes_to_explore:
            break

    # print(f'Checked {len(tree_nodes)} nodes in {time.time()-start:.2f} seconds for trivial state, but no success.')

    return False, path + [(action, new_length)]




if __name__=='__main__':
    # AK(2)
    relator1 = [1,1,-2,-2,-2]
    relator2 = [1,2,1,-2,-1,-2]
    max_length = 7

    def to_array(rel1, rel2, max_length):
        assert 0 not in rel1 and 0 not in rel2, "rel1 and rel2 must not be padded with zeros."
        assert max_length >= max(len(rel1), len(rel2)), "max_length must be > max of lengths of rel1, rel2"

        return  np.array(rel1 + [0]*(max_length-len(rel1)) + rel2 + [0]*(max_length-len(rel2)), 
                                dtype=np.int8)

    _, path = gs(presentation=np.array([]), rel1=relator1, rel2=relator2, max_length=max_length)
    print(path)

    # check that the path actually trivializes the initial state
    if path:
        state = np.array(relator1 + [0]*(max_length-len(relator1)) + 
                        relator2 + [0]*(max_length-len(relator2)), 
                                    dtype=np.int8)
        lengths = [len(relator1), len(relator2)]

        for action, _ in path[1:]:
            old_state = state.copy()
            state, lengths = ACMove(action, state, max_length, lengths, cyclical=False)

        print(f"Final state: {state}")