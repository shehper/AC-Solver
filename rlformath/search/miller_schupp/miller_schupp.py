"""
Applies a search algorithm of your choice to all presentations of Miller-Schupp series up to specified n and length(w).
Miller-Schupp presentations are labelled by an integer n >= 1 and a word w in two generators x and y with zero exponent sum on x.
MS(n, w) = <x, y | x^{-1} y^n x = y^{n+1}, x = w>
"""

import os
import argparse
import numpy as np
from itertools import product
from rlformath.envs.ac_env import simplify_relator

def generate_miller_schupp_presentations(n, max_w_len):
    """
    Generates Miller-Schupp presentations with fixed n >= 1 but all word lengths up to `max_w_len`.
    MS(n, w) = <x, y | x^{-1} y^n x = y^{n+1}, x = w>
    # TODO: Explain the reason behind fixing n but letting lenw vary?
    If two presentations are related by freely and cyclically reducing x^{-1} w, or through a cyclic permutation 
    of generators in x^{-1}w, we keep only one presentation.
    An example of the latter case: we keep only one of x^{-1} y x^{-1} y^2 x^2 and  x^{-1} y^2 x^2 x^{-1} y.

    Parameters: 
    n (int): n of Miller-Schupp series 
    max_w_len (an int): Maximum word-length of w of Miller-Schupp series"

    Returns:
    dict: A dictionary with the following structure:
        'lenw' (int): The length of word w.
        'presentations (list)': A list of presentations with fixed (n, length(w)).

    """
    assert n >= 1 and max_w_len >= 1, f"expect n >= 1 and max_w_len >=1 ; got n = {n}, max_w_len = {max_w_len}"

    rels = []
    nrel = 2 * max(2 * n + 3, max_w_len + 1) + 2
    rel1 = [-1] + [2] * n + [1] + [-2] * (n + 1) + [0] * (nrel - 2 * n - 3)

    seen = set()
    out = {}

    for search_len in range(1, max_w_len + 1):
        # iterate over all possible words of 1, 2, -1, -2 that have length = search_len
        for rel in product([1, 2, -1, -2], repeat=search_len):
            # only keep words where exponent sum of x is 0.
            if sum(x for x in rel if abs(x) == 1) != 0:
                continue

            # define rel2 = x^{-1}w -- the second relator of a Miller Schupp presentation.
            rel2 = np.array([-1] + list(rel), dtype=np.int8)
            # reduce x^{-1}w cyclically and freely by applying simplify(..., full=True) function from acenv.py
            # TODO: why is max_relator_length = search_len + 1?
            rel2, _ = simplify_relator(rel2, search_len + 1, cyclical=True, padded=False)

            rel2 = list(rel2)
            # discard the case where simplified word = x^{-1}, for example, if w was x x^{-1}.
            if rel2 == [-1]:
                continue

            lenw = len(rel2) - 1

            # if x^{-1} w is a cyclic permutation of x^{-1}w', only keep one of the two.
            if tuple(rel2) not in seen:
                for i in range(len(rel2)):
                    seen.add(tuple(rel2[i:] + rel2[:i]))
                rel2 += [0] * (nrel - len(rel2))

                if lenw not in out:
                    out[lenw] = []
                out[lenw] += [rel1 + rel2]

    return out

def write_list_to_text_file(list, filepath):
    with open(filepath, 'w') as f:
        for element in list:
            f.write(f"{element}\n")

def trivialize_miller_schupp_through_search(min_n, 
                                            max_n, 
                                            min_w_len, 
                                            max_w_len, 
                                            max_nodes_to_explore, 
                                            search_fn, 
                                            write_output_to_file=False):
    
    rels = {}

    for n in range(min_n, max_n + 1):
        rels[n] = generate_miller_schupp_presentations(n, max_w_len)

    solved_rels, unsolved_rels, solved_paths = [], [], []
    for n in range(min_n, max_n + 1):
        for lenw in range(min_w_len, max_w_len + 1):
            print(f"Applying {search_fn.__name__} to presentations of n = {n}, lenw = {lenw}")

            for pres in rels[n][lenw]:
                solved, path = search_fn(
                    presentation=pres, 
                    max_nodes_to_explore=max_nodes_to_explore, 
                    verbose=False, 
                    cyclically_reduce_after_moves=False
                )
                if solved:
                    solved_rels.append(pres)
                    solved_paths.append(path)
                else:
                    unsolved_rels.append(pres)

    if write_output_to_file:
        dirname = os.path.realpath(__file__)
        filename_base = f"n-{min_n}-to-{max_n}_lenw-{min_w_len}-to-{max_w_len}-{search_fn.__name__}"
        filepath_base = os.path.join(dirname, filename_base)
        write_list_to_text_file(list=solved_rels, filepath=filepath_base + "_solved")
        write_list_to_text_file(list=unsolved_rels, filepath=filepath_base + "_unsolved")
        write_list_to_text_file(list=solved_paths, filepath=filepath_base + "_paths")

    return solved_rels, unsolved_rels, solved_paths


if __name__ == "__main__":
    # TODO: change output file path (so that it lies in data) and names so that it specifies which search algorithm was used.
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--min-n",
            type=int,
            default=1,
            help="Minimum value of label n of Miller-Schupp series",
        )
        parser.add_argument(
            "--max-n",
            type=int,
            default=7,
            help="Maximum value of label n of Miller-Schupp series",
        )
        parser.add_argument(
            "--min-w-len", 
            type=int, 
            default=1, 
            help="Minimum word-length of w of Miller-Schupp series",
        )
        parser.add_argument(
            "--max-w-len", 
            type=int, 
            default=7, 
            help="Maximum word-length of w of Miller-Schupp series",
        )
        parser.add_argument(
            "--max-nodes-to-explore", 
            type=int, 
            default=int(1e6), 
            help="Maximum number of nodes to explore during tree search.",
        )
        parser.add_argument(
            "--search-algorithm",
            type=str,
            default="greedy",
            help="the name of this experiment",
        )
        args = parser.parse_args()
        return args
    
    args = parse_args()
    assert args.search_algorithm in ["greedy", "bfs"], f"expect search-algorithm to be greedy or bfs; got {args.search_algorithm}"

    if args.search_algorithm == "greedy":
        from rlformath.search.greedy import greedy_search
        search_fn = greedy_search
    elif args.search_algorithm == "bfs":
        from rlformath.search.breadth_first import bfs
        search_fn = bfs
    else:
        raise ValueError(f"Unsupported search algorithm: {args.search_algorithm}; expect greedy or bfs")

    solved_rels, unsolved_rels, solved_paths = trivialize_miller_schupp_through_search(
        min_n = args.min_n,
        max_n = args.max_n,
        min_w_len = args.min_w_len,
        max_w_len=args.max_w_len,
        max_nodes_to_explore=args.max_nodes_to_explore,
        search_fn=search_fn,
        write_output_to_file=True,
    )

    print(solved_rels)
    print(unsolved_rels)
    print(solved_paths)