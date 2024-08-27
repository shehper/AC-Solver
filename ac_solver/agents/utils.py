"""
This file contains some helper functions for PPO agents. 

"""

from importlib import resources
from ast import literal_eval


def load_initial_states_from_text_file(states_type):
    """
    Loads initial presentations from a text file based on the specified state type. The presentations
    are sorted by their hardness.

    Parameters:
    states_type (str): The type of states to load. Must be either "solved" or "all".

    Returns:
    list: A list of presentations loaded from the text file.

    Raises:
    AssertionError: If states_type is not "solved" or "all".
    """
    assert states_type in ["solved", "all"], "states_type must be 'solved' or 'all'"

    file_name_prefix = "greedy_solved" if states_type == "solved" else "all"
    file_name = f"{file_name_prefix}_presentations.txt"
    with resources.open_text("ac_solver.search.miller_schupp.data", file_name) as file:
        initial_states = [literal_eval(line.strip()) for line in file]

    print(f"Loaded {len(initial_states)} presentations from {file_name}.")
    return initial_states
