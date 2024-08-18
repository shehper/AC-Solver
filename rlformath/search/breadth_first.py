"""This file contains an actual BFS algorithm. It uses a queue instead of a priority queue to search through the space of presentations."""

from rlformath.envs.ac_env import ACMove
import time
from collections import deque 
import numpy as np

# TODO: include the ability to save path.

def to_array(rel1, rel2, max_length):
    assert 0 not in rel1 and 0 not in rel2, "rel1 and rel2 must not be padded with zeros."
    assert max_length >= max(len(rel1), len(rel2)), "max_length must be > max of lengths of rel1, rel2"

    return  np.array(rel1 + [0]*(max_length-len(rel1)) + rel2 + [0]*(max_length-len(rel2)), 
                             dtype=np.int8)

def bfs(rels=np.array([]), rel1=None, rel2=None, max_length=None, 
        cap_num_nodes=True, max_num_nodes=int(1e6), verbose=True,
        full_simplify_moves=False):
    # TODO: This is hard to work with. Let rels be either a numpy array or a tuple of
    # (rel1, rel2, max_length).
    """Takes either a numpy array/list 'rels' of shape (2*max_length,) or 
    takes rel1, rel2, max_length and converts them into this shape."""
    rels = np.array(rels, dtype=np.int8) # so that input may be a list or a tuple
    if rels.any():
        initial_state = rels
        max_length = len(rels)//2
        lengths = [np.count_nonzero(rels[:max_length]), np.count_nonzero(rels[max_length:])]
    else:
        assert rel1 and rel2 and max_length, "Either give rels or all rel1, rel2, and max_length"
        initial_state = to_array(rel1, rel2, max_length) 
        lengths = [len(rel1), len(rel2)]
    
    # add to a queue, keeping track of path length to initial state
    state_tup = tuple(initial_state)
    init_path = [(0, sum(lengths))]
    to_explore = deque([(state_tup, init_path)]) # 
    lengths = [np.count_nonzero(initial_state[:max_length]), np.count_nonzero(initial_state[max_length:])]
    min_length = sum(lengths)

    # a set containing states that have already been seen
    tree_nodes = set() 
    tree_nodes.add(tuple(initial_state)) 

    start = time.time() 
    while (to_explore):
        state_tuple, path = to_explore.popleft()
        state = np.array(state_tuple, dtype=np.int8) # convert tuple to state
        lengths = [np.count_nonzero(rels[:max_length]), np.count_nonzero(rels[max_length:])]

        for action in range(1, 13):    
            new_state, new_lengths = ACMove(move_id=action, 
                                            presentation=state, 
                                            max_relator_length=max_length, 
                                            lengths=lengths, 
                                            cyclical=full_simplify_moves) 
            state_tup, new_length = tuple(new_state), sum(new_lengths)
            
            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f'New minimal length found: {min_length}')

            if new_length == 2:
                #if verbose:
                    #print(f"Found {new_state[0:1], new_state[max_length:max_length+1]} 
                    # after exploring {len(tree_nodes)-len(to_explore)} nodes")
                    # print(f"Path to this state:  {path + [(action, new_length)]}")
                    #print(f"Total path length: {len(path)+1}")
                return True, path + [(action, new_length)]

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                to_explore.append((state_tup, path + [(action, new_length)]))

        if cap_num_nodes and len(tree_nodes) >= max_num_nodes:
            break

    print(f'Checked {len(tree_nodes)} nodes in {time.time()-start:.2f} seconds for trivial state, but no success.')

    return False, None



if __name__=='__main__':
    # AK(2)
    relator1 = [1,1,-2,-2,-2]
    relator2 = [1,2,1,-2,-1,-2]
    max_length = 7

    ans, path = bfs(rels=np.array([]), rel1=relator1, rel2=relator2, max_length=max_length)
    print(ans)
    print(path)
    if path:
        state = np.array(relator1 + [0]*(max_length-len(relator1)) + 
                        relator2 + [0]*(max_length-len(relator2)), 
                                    dtype=np.int8)
        lengths = [len(relator1), len(relator2)]

        for action, _ in path[1:]:
            old_state = state.copy()
            state, lengths = ACMove(move_id=action,
                                    presentation=state,
                                    max_relator_length=max_length,
                                    lengths=lengths,
                                    cyclical=False)

        print(f"Final state: {state}")
        # print(f"Does initial state equal final state: Expected: {True}; got: {np.array_equal(state, )}")