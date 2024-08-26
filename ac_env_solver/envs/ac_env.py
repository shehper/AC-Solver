"""
Implements Andrews-Curtis (AC) moves for presentations with two generators.
"""

from dataclasses import dataclass, field
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


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
    presentation, max_relator_length, lengths_of_words, cyclical=True
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
        simplified_relator, length_i = simplify_relator(
            relator=presentation[i * max_relator_length : (i + 1) * max_relator_length],
            max_relator_length=max_relator_length,
            cyclical=cyclical,
            padded=True,
        )

        presentation[i * max_relator_length : (i + 1) * max_relator_length] = (
            simplified_relator
        )
        lengths_of_words[i] = length_i

    return presentation, lengths_of_words


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


def concatenate_relators(presentation, max_relator_length, i, j, sign, lengths):
    """
    Given a presentation <r_0, r_1>, returns a new presentation where r_i is replaced by r_i r_j^{sign}.

    Parameters:
    presentation: A Numpy Array representing a presentation
    max_relator_length: An int. Maximum length the concatenated relator is allowed to have.
                        If length of the concatenated relator (after simplification) is greater than this integer,
                        the original presentation is returned without any changes.
                        The only simplifications applied are free reductions and not cyclical reductions as the latter
                        correspond to conjugations on a given word.
    i: 0 or 1, index of the relator to change.
    j: 0 or 1, but not equal to i.
    sign: +1 or -1, whether to invert r_j before concatenation.
    lengths: A list of lengths of words in presentation.

    Returns:
    (resultant_presentation, lengths_of_resultant_presentations)
    resultant_presentation is the presentation with r_i possibly replaced with r_i r_j^{sign}.
    lengths_of_resultant_presentations is the list of lengths of words in the resultant presentation.
    """
    assert all(
        [
            i in [0, 1],
            j in [0, 1],
            i == 1 - j,
        ]
    ), f"expect i and j to be 0 or 1 and i != j; got i = {i}, j = {j}"

    assert sign in [1, -1], f"expect sign to be +1 or -1, received {sign}"

    # get r_i
    presentation = presentation.copy()
    relator1 = presentation[i * max_relator_length : (i + 1) * max_relator_length]

    # get r_j or r_j^{-1} depending on sign
    # TODO: really need to understand this
    if sign == 1:
        relator2 = presentation[j * max_relator_length : (j + 1) * max_relator_length]
    elif j:
        relator2 = -presentation[
            (j + 1) * max_relator_length - 1 : j * max_relator_length - 1 : -1
        ]
    else:
        relator2 = -presentation[max_relator_length - 1 :: -1]

    relator1_nonzero = relator1[relator1 != 0]
    relator2_nonzero = relator2[relator2 != 0]

    len1 = len(relator1_nonzero)
    len2 = len(relator2_nonzero)

    acc = 0
    while (
        acc < min(len1, len2) and relator1_nonzero[-1 - acc] == -relator2_nonzero[acc]
    ):
        acc += 1

    new_size = len1 + len2 - 2 * acc

    if new_size <= max_relator_length:
        lengths[i] = new_size
        presentation[i * max_relator_length : i * max_relator_length + len1 - acc] = (
            relator1_nonzero[: len1 - acc]
        )
        presentation[
            i * max_relator_length + len1 - acc : i * max_relator_length + new_size
        ] = relator2_nonzero[acc:]
        presentation[
            i * max_relator_length + new_size : (i + 1) * max_relator_length
        ] = 0

    return presentation, lengths


def conjugate(presentation, max_relator_length, i, j, sign, lengths):
    """
    Given a presentation <r_0, r_1>, returns a new presentation where r_i is replaced by x_j^{sign} r_i x_j^{-sign}.

    Parameters:
    presentation: A Numpy Array representing a presentation
    max_relator_length: An int. Maximum length the concatenated relator is allowed to have.
                        If length of the concatenated relator (after simplification) is greater than this integer,
                        the original presentation is returned without any changes.
                        The only simplifications applied are free reductions and not cyclical reductions as the latter
                        correspond to conjugations on a given word.
    i: 0 or 1, index of the relator to change.
    j: 1 or 2, index of the generator to conjugate with.
    sign: +1 or -1, whether to invert x_j before concatenation.
    lengths: A list of lengths of words in presentation.

    Returns:
    (resultant_presentation, lengths_of_resultant_presentations)
    resultant_presentation is the presentation with r_i possibly replaced with x_j^{sign} r_i x_j^{-sign}.
    lengths_of_resultant_presentations is the list of lengths of words in the resultant presentation.

    """
    # TODO: perhaps i and j should be more uniformly both in [0, 1].
    assert all(
        [i in [0, 1], j in [1, 2]]
    ), f"expect i to be 0 and 1 and j to be 1 or 2; got i = {i}, j = {j}"

    assert sign in [1, -1], f"expect sign to be +1 or -1, received {sign}"

    presentation = presentation.copy()
    relator = presentation[i * max_relator_length : (i + 1) * max_relator_length]
    relator_nonzero = relator[relator.nonzero()]
    relator_size = len(relator_nonzero)

    # get the generator that is to be appended on the left
    generator = sign * j

    # TODO: again here, it will be good to use simplify_relator

    # check whether we will need to cancel any generators at the beginning and at the end
    start_cancel = 1 if relator_nonzero[0] == -generator else 0
    end_cancel = 1 if relator_nonzero[-1] == generator else 0

    # get the size of the resultant relator after cancellation
    new_size = relator_size + 2 - 2 * (start_cancel + end_cancel)

    # update lengths and presentation
    if new_size <= max_relator_length:
        lengths = lengths.copy()
        lengths[i] = new_size

        presentation[
            i * max_relator_length
            + 1
            - start_cancel : i * max_relator_length
            + 1
            + relator_size
            - 2 * start_cancel
            - end_cancel
        ] = relator_nonzero[start_cancel : relator_size - end_cancel]

        if not start_cancel:
            presentation[i * max_relator_length] = generator

        if not end_cancel:
            presentation[
                i * max_relator_length + relator_size + 1 - 2 * start_cancel
            ] = -generator

        if start_cancel and end_cancel:
            presentation[
                i * max_relator_length
                + new_size : i * max_relator_length
                + new_size
                + 2
            ] = 0

    return presentation, lengths


def ACMove(move_id, presentation, max_relator_length, lengths, cyclical=True):
    """
    Applies an AC move (concatenation or conjugation) to a presentation and returns the resultant presentation.
    The move to apply and the relator it is applied to are decided by move_id.

    Parameters:
    move_id: An int in range [1, 12] (both inclusive), deciding which AC move to apply.
            Odd values affect r_1; even values affect r_0.
            The complete mappling between move_id and moves is as below:
            1. r_1 --> r_1 r_0
            2. r_0 --> r_0 r_1^{-1}
            3. r_1 --> r_1 r_0^{-1}
            4. r_0 --> r_0 r_1
            5: r_1 --> x_0^{-1} r_1 x_0
            6: r_0 ---> x_1^{-1} r_0 x_1
            7: r_1 --> x_1^{-1} r_1 x_1
            8: r_0 ---> x_0 r_0 x_0^{-1}
            9: r_1 --> x_0 r_1 x_0^{-1}
            10: r_0 --> x_1 r_0 x_1^{-1}
            11: r_1 --> x_1 r_1 x_1^{-1}
            12: r_0 --> x_0^{-1} r_0 x_0
    presentation: A NumPy Array representation the input presentation.
    max_relator_length: The maximum length a relator is allowed to take.
                        If the application of an AC move results in a relator with length larger than max_relator_length,
                        the original presentation is returned.
    lengths: A list of lengths of words in the presentation.
    cyclical: A bool; whether to cyclically reduce words in the resultant presentation or not.
    """

    assert move_id in range(
        1, 13
    ), f"Expect n to be in range 1-12 (both inclusive); got {move_id}"

    if move_id in range(1, 5):
        i = move_id % 2
        j = 1 - i
        sign_parity = ((move_id - i) // 2) % 2
        sign = (-1) ** sign_parity
        move = concatenate_relators
    elif move_id in range(5, 13):
        i = move_id % 2
        jp = ((move_id - i) // 2) % 2  # = 0 or 1
        sign_parity = ((move_id - i - 2 * jp) // 4) % 2
        j = jp + 1  # = 1 or 2
        sign = (-1) ** sign_parity
        move = conjugate

    presentation, lengths = move(
        presentation=presentation,
        max_relator_length=max_relator_length,
        i=i,
        j=j,
        sign=sign,
        lengths=lengths,
    )

    # TODO: simplify_presentation seems to do something non-trivial even when
    # cyclical=False. I ran into trouble by putting an `if cyclical==False` cond
    # before the next lines of code.
    # This is confusing because I thought cojugate and concatenate_relators
    # already do the cyclical=False simplification.

    # TODO: cyclical should probably be called cylically_reduce.
    presentation, lengths = simplify_presentation(
        presentation=presentation,
        max_relator_length=max_relator_length,
        lengths_of_words=lengths,
        cyclical=cyclical,
    )

    return presentation, lengths


@dataclass
class ACEnvConfig:
    max_relator_length: int = 7
    init_presentation: np.ndarray = field(default_factory=lambda: np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0]))
    max_count_steps: int = 1000
    use_supermoves: bool = False

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            max_relator_length=config_dict.get("max_relator_length", cls().max_relator_length),
            init_presentation=np.array(config_dict.get("init_presentation", cls().init_presentation)),
            max_count_steps=config_dict.get("max_count_steps", cls().max_count_steps),
            use_supermoves=config_dict.get("use_supermoves", cls().use_supermoves)
        )

class ACEnv(Env):
    # TODO: I think I forgot to mention in the paper that if an episode terminates successfully,
    # we give a large maximum reward.
    def __init__(self, config: ACEnvConfig = ACEnvConfig()):
        self.n_gen = 2
        self.max_relator_length = config.max_relator_length
        # 3 names for the same thing: state, init_presentation, initial_state. Must consolidate.
        self.state = config.init_presentation
        self.initial_state = np.copy(config.init_presentation)
        self.max_count_steps = config.max_count_steps  # call it horizon_length
        if config.use_supermoves:
            raise NotImplementedError(
                "ACEnv with supermoves is not yet implemented in this library."
            )

        assert (
            len(self.state) == self.n_gen * self.max_relator_length
        ), f"The total length of init_presentation = {len(config.init_presentation)} must be equal \
             to  {2 * self.max_relator_length}."
        self.count_steps = 0
        self.lengths = [
            np.count_nonzero(
                self.state[
                    i * self.max_relator_length : (i + 1) * self.max_relator_length
                ]
            )
            for i in range(self.n_gen)
        ]
        self.max_reward = self.max_count_steps * self.max_relator_length * self.n_gen
        self.action_space = Discrete(12)
        self.actions = []

        low = np.ones(self.max_relator_length * self.n_gen, dtype=np.int8) * (
            -self.n_gen
        )
        high = np.ones(self.max_relator_length * self.n_gen, dtype=np.int8) * (
            self.n_gen
        )
        self.observation_space = Box(low, high, dtype=np.int8)

    def step(self, action):
        # action is in [0,11] but the input to ACMove is in [1,12] so we give action+1 as input to ACMove.
        self.actions.append(int(action + 1))
        # if action + 1 is a supermove, apply all actions in the supermove
        self.state, self.lengths = ACMove(
            action + 1, self.state, self.max_relator_length, self.lengths
        )

        done = sum(self.lengths) == 2
        reward = self.max_reward * done - sum(self.lengths) * (1 - done)

        self.count_steps += 1
        truncated = self.count_steps >= self.max_count_steps

        return (
            self.state,
            reward,
            done,
            truncated,
            {"actions": self.actions.copy()} if done else {},
        )

    def reset(self, *, seed=None, options=None):
        self.state = (
            np.copy(options["starting_state"])
            if options and "starting_state" in options
            else np.copy(self.initial_state)
        )
        self.lengths = [
            np.count_nonzero(
                self.state[
                    i * self.max_relator_length : (i + 1) * self.max_relator_length
                ]
            )
            for i in range(self.n_gen)
        ]
        self.count_steps = 0
        self.actions = []
        return self.state, {}

    def render(self):
        pass


# all variables here: rel, nrel, full, ngen, lengths, rels, i, j, sign,
# nrel is length of relator: probably should call it max_relator_len
# rel: relator
# rels: presentation
# lengths: relator_lengths # TODO: is this actually needed?
# i, j:
# TODO: specify in many places that what's called AC here are really AC-prime moves.
# TODO: change lengths to word_lengths everywhere.
# TODO: lengths should probably not be input or outputted into all of the functions anyway.
# There should be a separate function that computes lengths of words given a presentation.
# functions: len_words, simplify, full_simplify, is_trivial, trivial_states, concat, conjugate
# TODO: len_words should just be replaced with np.count_nonzero() everywhere in the codebase.
# simplify should be called simplify_relator; I don't know if we ever need padded=False so may well remove that
# also maybe full should be called 'cyclic' or something like that.

# TODO: there should be tests for simplify, full_simplify, is_trivial, trivial_states, concat and conjugate.
# TODO: fix indentation. Why is it half as usual here?
# computes number of nonzero elements of a Numpy array
# TODO: max_relator_length should be needed only in conjugate and concat, I think.
# TODO: lengths_of_words should not be given as a parameter anywhere. We should just compute length of word when we need to.
# Perhaps there can be a separate function for that.