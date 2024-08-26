"""
This file contains various helper functions for agents. 

"""

import math
import numpy as np
from importlib import resources
from ast import literal_eval


def convert_relators_to_presentation(relator1, relator2, max_relator_length):
    """
    Converts two lists representing relators into a single numpy array, padding each relator with zeros
    to match the specified maximum length.

    Parameters:
    relator1 (list of int): The first relator, must not contain zeros.
    rel2 (list of int): The second relator, must not contain zeros.
    max_relator_length (int): The maximum allowed length for each relator.

    Returns:
    np.ndarray: A numpy array of dtype int8, containing the two relators concatenated and zero-padded to max_length.
    """

    # Ensure relators do not contain zeros and max_relator_length is sufficient
    assert (
        0 not in relator1 and 0 not in relator2
    ), "relator1 and relator2 must not be padded with zeros."
    assert max_relator_length >= max(
        len(relator1), len(relator2)
    ), "max_relator_length must be greater than or equal to the lengths of relator1 and rel2."
    assert isinstance(relator1, list) and isinstance(
        relator2, list
    ), f"got types {type(relator1)} for relator1 and {type(relator2)} for relator2"

    padded_relator1 = relator1 + [0] * (max_relator_length - len(relator1))
    padded_relator2 = relator2 + [0] * (max_relator_length - len(relator2))

    return np.array(padded_relator1 + padded_relator2, dtype=np.int8)


def change_max_relator_length_of_presentation(presentation, new_max_length):
    """
    Adjusts the maximum length of the relators in a given presentation by reformatting it
    with a new specified maximum length.

    Parameters:
    presentation (Numpy Array): The current presentation as a list, where relators are concatenated and padded with zeros.
    new_max_length (int): The new maximum length for each relator in the presentation.

    Returns:
    Numpy Array: The new presentation with relators adjusted to the specified maximum length.
    """

    old_max_length = len(presentation) // 2

    first_word_length = np.count_nonzero(presentation[:old_max_length])
    second_word_length = np.count_nonzero(presentation[old_max_length:])

    relator1 = presentation[:first_word_length]
    relator2 = presentation[old_max_length : old_max_length + second_word_length]

    new_presentation = convert_relators_to_presentation(
        relator1=relator1, relator2=relator2, max_relator_length=new_max_length
    )
    return new_presentation


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
    with resources.open_text(
        "ac_env_solver.search.miller_schupp.data", file_name
    ) as file:
        initial_states = [literal_eval(line.strip()) for line in file]

    print(f"Loaded {len(initial_states)} presentations from {file_name}.")
    return initial_states


def get_curr_lr(n_update, lr_decay, warmup, max_lr, min_lr, total_updates):
    """
    Calculates the current learning rate based on the update step, learning rate decay schedule,
    warmup period, and other parameters.

    Parameters:
    n_update (int): The current update step (1-indexed).
    lr_decay (str): The type of learning rate decay to apply ("linear" or "cosine").
    warmup (float): The fraction of total updates to be used for the learning rate warmup.
    max_lr (float): The maximum learning rate.
    min_lr (float): The minimum learning rate.
    total_updates (int): The total number of updates.

    Returns:
    float: The current learning rate.

    Raises:
    NotImplementedError: If an unsupported lr_decay type is provided.
    """
    # Convert to 0-indexed for internal calculations
    n_update -= 1
    total_updates -= 1

    # Calculate the end of the warmup period
    warmup_period_end = total_updates * warmup

    if warmup_period_end > 0 and n_update <= warmup_period_end:
        lrnow = max_lr * n_update / warmup_period_end
    else:
        if lr_decay == "linear":
            slope = (max_lr - min_lr) / (warmup_period_end - total_updates)
            intercept = max_lr - slope * warmup_period_end
            lrnow = slope * n_update + intercept

        elif lr_decay == "cosine":
            cosine_arg = (
                (n_update - warmup_period_end)
                / (total_updates - warmup_period_end)
                * math.pi
            )
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplementedError(
                "Only 'linear' and 'cosine' lr-schedules are available."
            )

    return lrnow
