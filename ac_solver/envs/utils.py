"""
This file contains various helper functions concerning balanced presentations and AC Environment.

(For the purpose of this project, a balanced presentation is a NumPy array of even length, whose:
i. elements are all 1, -1, 2, -2, or 0.
ii. zeros in each half of the array are padded to the right end.
)
"""

import numpy as np
from collections import deque


def is_array_valid_presentation(array):
    """
    Checks whether a given Numpy Array is a valid presentation or not.
    An array is a valid presentation with two words if each half has all zeros padded to the right.
    That is, [1, 2, 0, 0, -2, -1, 0, 0] is a valid presentation, but [1, 0, 2, 0, -2, -1, 0, 0] is not.
    And if each word has nonzero length, i.e. [0, 0, 0, 0] and [1, 2, 0, 0] are invalid.

    Parameters:
    presentation: A Numpy Array

    Returns: True / False
    """

    # for two generators and relators, the length of the presentation should be even.
    assert isinstance(
        array, (list, np.ndarray)
    ), f"array must be a list or a numpy array, got {type(array)}"
    if isinstance(array, list):
        array = np.array(array)

    is_length_valid = len(array) % 2 == 0

    max_relator_length = len(array) // 2

    first_word_length = np.count_nonzero(array[:max_relator_length])
    second_word_length = np.count_nonzero(array[max_relator_length:])

    is_first_word_valid = (array[first_word_length:max_relator_length] == 0).all()
    is_second_word_valid = (array[max_relator_length + second_word_length :] == 0).all()

    # for a presentation to be valid, each word should have length >= 1 and it should have all the zeros padded to the right.
    is_valid = all(
        [
            is_length_valid,
            first_word_length > 0,
            second_word_length > 0,
            is_first_word_valid,
            is_second_word_valid,
        ]
    )

    return is_valid


def is_presentation_trivial(presentation):
    """
    Checks whether a given presentation is trivial or not. (Assumes two generators and relators)
    For two generators, there are eight possible trivial presentations: <x, y>, <y, x> or any other
    obtained by replacing x --> x^{-1} and / or y --> y^{-1}.

    Parameters:
    presentation: A Numpy Array

    """
    # presentation should be valid
    if not is_array_valid_presentation(presentation):
        return False

    # each word length should be exactly 1
    max_relator_length = len(presentation) // 2
    for i in range(2):
        if (
            np.count_nonzero(
                presentation[i * max_relator_length : (i + 1) * max_relator_length]
            )
            != 1
        ):
            return False

    # if each word length is 1, each generator or its inverse should appear exactly once,
    # i.e. non zero elements, after sorting, should equal to np.array([1, 2]).
    non_zero_elements = abs(presentation[presentation != 0])
    non_zero_elements.sort()
    is_trivial = np.array_equal(non_zero_elements, np.arange(1, 3))
    return is_trivial


# Returns the set of trivial states of the right length (i.e. 8 states )
def generate_trivial_states(max_relator_length):
    r"""
    Generate Numpy Arrays of trivial states for a given max_relator_length.
    e.g. if max_relator_length = 3, one trivial state is [1, 0, 0, 2, 0, 0].
    There are 8 trivial states in total: (x^{\pm 1}, y^{\pm 1}) and (y^{\pm 1}, x^{\pm 1})

    Parameters:
    max_relator_length: An int

    Returns:
    A numpy array of shape (8, 2 * max_relator_length), containing eight trivial states.
    """
    states = []
    for i in [1, 2]:
        for sign1 in [-1, 1]:
            for sign2 in [-1, 1]:
                states += [
                    [sign1 * i]
                    + [0] * (max_relator_length - 1)
                    + [sign2 * (3 - i)]
                    + [0] * (max_relator_length - 1)
                ]

    return np.array(states)


def convert_relators_to_presentation(relator1, relator2, max_relator_length):
    """
    Converts two lists representing relators into a single numpy array, padding each relator with zeros
    to match the specified maximum length.

    Parameters:
    relator1 (list of int): The first relator, must not contain zeros.
    relator2 (list of int): The second relator, must not contain zeros.
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


def simplify_relator(relator, max_relator_length, cyclical=False, padded=True):
    """
    Simplifies a relator by removing neighboring inverses. For example, if input is x^2 y y^{-1} x, the output will be x^3.

    Parameters:
    relator (numpy array): An array representing a word of generators.
                           Expected form is to have non-zero elements to the left and any padded zeros to the right.
                           For example, [-1, -1, 2, 2, 1, 0, 0] represents the word x^{-2} y^2 x
    max_relator_length (int): Upper bound on the length of the simplified relator.
                              If, after simplification, the relator length > max_relator_length, an assertion error is given.
                              This bound is placed so as to make the search space finite. Say,
    cyclical (bool): A bool to specify whether to remove inverses on opposite ends of the relator.
                     For example, when true, the function will reduce x y x^{-1} to y.
                     This is equivalent to conjugation by a word.
    padded (bool): A bool to specify whether to pad the output with zeros on the right end.
                   If True, the output array has length = max_relator_length, with number of padded zeros
                   equal to the difference of max_relator_length and word length of the simplified relator.

    Returns: (simplified_relator, relator_length)
    simplified_relator is a Numpy array representing the simplified relator
    relator_length is the length of simplified word.
    """

    assert isinstance(relator, np.ndarray), "expect relator to be a numpy array"

    # number of nonzero entries
    relator_length = np.count_nonzero(relator)
    if len(relator) > relator_length:
        assert (
            relator[relator_length:] == 0
        ).all(), "expect all zeros to be at the right end"

    # loop over the array and remove inverses
    pos = 0
    while pos < relator_length - 1:
        if relator[pos] == -relator[pos + 1]:
            indices_to_remove = [pos, pos + 1]
            relator = np.delete(relator, indices_to_remove)
            relator_length -= 2
            if pos:
                pos -= 1
        else:
            pos += 1

    # if cyclical, also remove inverses from the opposite ends
    if cyclical and relator_length > 0:
        pos = 0
        while relator[pos] == -relator[relator_length - pos - 1]:
            pos += 1
        if pos:
            indices_to_remove = np.concatenate(
                [np.arange(pos), relator_length - 1 - np.arange(pos)]
            )
            relator = np.delete(relator, indices_to_remove)
            relator_length -= 2 * pos

    # if padded, pad zeros to the right so that the output array has length = max_relator_length
    if padded:
        relator = np.pad(relator, (0, max_relator_length - len(relator)))

    assert (
        max_relator_length >= relator_length
    ), "Increase max length! Length of simplified word \
                                                  is bigger than maximum allowed length."

    return relator, relator_length


def simplify_presentation(
    presentation, max_relator_length, lengths_of_words, cyclical=True, use_deque=True
):
    """
    Simplifies a presentation by simplifying each of its relators. (See `simplify_relator` for more details.)

    Parameters:
    presentation: A Numpy Array
    max_relator_length: maximum length a simplified relator is allowed to take
    lengths_of_words: A list containing length of each word.

    Returns:
    (simplified_presentation, lengths_of_simplified_words)
    simplified_presentation is a Numpy Array
    lengths_of_simplified_words is a list of lengths of simplified words.
    """

    presentation = np.array(presentation)  # TODO: Is this necessary?
    assert is_array_valid_presentation(
        presentation
    ), f"{presentation} is not a valid presentation. Expect all zeros to be padded to the right."

    lengths_of_words = lengths_of_words.copy()
    for i in range(2):
        if use_deque:
            simplified_relator, length_i = simplify_relator_with_deque(
                relator=presentation[
                    i * max_relator_length : (i + 1) * max_relator_length
                ],
                max_relator_length=max_relator_length,
                cyclical=cyclical,
                padded=True,
            )
        else:
            simplified_relator, length_i = simplify_relator(
                relator=presentation[
                    i * max_relator_length : (i + 1) * max_relator_length
                ],
                max_relator_length=max_relator_length,
                cyclical=cyclical,
                padded=True,
            )

        presentation[i * max_relator_length : (i + 1) * max_relator_length] = (
            simplified_relator
        )
        lengths_of_words[i] = length_i

    return presentation, lengths_of_words


def simplify_relator_with_deque(
    relator, max_relator_length, cyclical=False, padded=True
):
    """
    Simplifies a relator by removing neighboring inverses. For example, if input is x^2 y y^{-1} x, the output will be x^3.

    Parameters:
    relator (numpy array): An array representing a word of generators.
                           Expected form is to have non-zero elements to the left and any padded zeros to the right.
                           For example, [-1, -1, 2, 2, 1, 0, 0] represents the word x^{-2} y^2 x
    max_relator_length (int): Upper bound on the length of the simplified relator.
                              If, after simplification, the relator length > max_relator_length, an assertion error is given.
                              This bound is placed so as to make the search space finite. Say,
    cyclical (bool): A bool to specify whether to remove inverses on opposite ends of the relator.
                     For example, when true, the function will reduce x y x^{-1} to y.
                     This is equivalent to conjugation by a word.
    padded (bool): A bool to specify whether to pad the output with zeros on the right end.
                   If True, the output array has length = max_relator_length, with number of padded zeros
                   equal to the difference of max_relator_length and word length of the simplified relator.

    Returns: (simplified_relator, relator_length)
    simplified_relator is a Numpy array representing the simplified relator
    relator_length is the length of simplified word.
    We can use a double-ended queue to finish this task.
    """
    assert isinstance(relator, np.ndarray), "expect relator to be a numpy array"
    relator_deque = deque()
    relator_length = np.count_nonzero(relator)
    relator_padding_length = len(relator)
    for i in range(relator_length):
        if relator_deque and relator_deque[-1] + relator[i] == 0:
            relator_deque.pop()
        else:
            relator_deque.append(relator[i])
    if cyclical:
        while relator_deque and relator_deque[0] + relator_deque[-1] == 0:
            relator_deque.popleft()
            relator_deque.pop()
    relator = np.array(relator_deque)
    relator_length = len(relator)
    if padded:
        relator = np.pad(relator, (0, max_relator_length - len(relator)))
    else:
        relator = np.pad(relator, (0, relator_padding_length - len(relator)))
    return relator, relator_length
