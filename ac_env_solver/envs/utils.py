import numpy as np

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