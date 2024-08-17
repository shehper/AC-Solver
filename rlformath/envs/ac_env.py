"""
Implements Andrews-Curtis (AC) moves for presentations with two generators.
"""

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

# all variables here: rel, nrel, full, ngen, lengths, rels, i, j, sign, 
# nrel is length of relator: probably should call it max_relator_len
# rel: relator
# rels: presentation
# lengths: relator_lengths # TODO: is this actually needed?
# i, j: 

# functions: len_words, simplify, full_simplify, is_trivial, trivial_states, concat, conjugate
# TODO: len_words should just be replaced with np.count_nonzero() everywhere in the codebase. 
# simplify should be called simplify_relator; I don't know if we ever need padded=False so may well remove that
# also maybe full should be called 'cyclic' or something like that.
# 

# TODO: there should be tests for simplify, full_simplify, is_trivial, trivial_states, concat and conjugate.

# TODO: fix indentation. Why is it half as usual here?
# computes number of nonzero elements of a Numpy array
def len_words(relators):
    # TODO: this function should be removed.
    return np.count_nonzero(relators)

# TODO: max_relator_length should be needed only in conjugate and concat, I think. 
# TODO: lengths_of_words should not be given as a parameter anywhere. We should just compute length of word when we need to. 
# Perhaps there can be a separate function for that. 

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
    simplified_relator is the simplified relator, and relator_length is the length of this word.
    """
    
    assert isinstance(relator, np.ndarray), "expect relator to be a numpy array"

    # number of nonzero entries
    relator_length = np.count_nonzero(relator)
    if len(relator) > relator_length:
        assert relator[relator_length:] == 0, "expect all zeros to be at the right end"
    
    # loop over the array and remove inverses
    pos = 0
    while pos < relator_length - 1:
        if relator[pos] == - relator[pos + 1]:
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
            indices_to_remove = np.concatenate([np.arange(pos), relator_length - 1 - np.arange(pos)])
            relator = np.delete(
                relator, indices_to_remove
            )
            relator_length -= 2 * pos

    # if padded, pad zeros to the right so that the output array has length = max_relator_length
    if padded:
        relator = np.pad(relator, (0, max_relator_length - len(relator)))

    assert max_relator_length >= relator_length, "Increase max length! Length of simplified word \
                                                  is bigger than maximum allowed length."

    return  relator, relator_length


# Simplifies each relator in a list of relator.
def simplify_presentation(presentation, max_relator_length, lengths_of_words, cyclical=True):
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
    
    presentation = np.array(presentation) # TODO: are we allowing presentation to be a list?
    assert is_presentation_valid(presentation), f"{presentation} is not a valid presentation. Expect all zeros to be padded to the right." 
    
    lengths_of_words = lengths_of_words.copy()

    for i in range(2): # assuming only two generators / words for now; TODO: generalize
        presentation[i * max_relator_length : (i + 1) * max_relator_length], lengths_of_words[i] = simplify_relator(
            presentation[i * max_relator_length : (i + 1) * max_relator_length], max_relator_length, cyclical=cyclical, padded=True
        )

    return presentation, lengths_of_words


def is_presentation_valid(presentation):
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
    is_length_valid = len(presentation) % 2 == 0

    max_relator_length = len(presentation) // 2

    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])

    is_first_word_valid = (presentation[first_word_length : max_relator_length] == 0).all()
    is_second_word_valid = (presentation[max_relator_length + second_word_length :] == 0).all()

    # for a presentation to be valid, each word should have length >= 1 and it should have all the zeros padded to the right.
    is_valid = all([
        is_length_valid,
        first_word_length > 0,
        second_word_length > 0,
        is_first_word_valid,
        is_second_word_valid
    ])

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
    if not is_presentation_valid(presentation):
        return False
    
    # each word length should be exactly 1
    max_relator_length = len(presentation) // 2
    for i in range(2):
        if np.count_nonzero(presentation[i * max_relator_length : (i + 1) * max_relator_length]) != 1:
            return False

    # if each word length is 1, each generator or its inverse should appear exactly once,
    # i.e. non zero elements, after sorting, should equal to np.array([1, 2]).
    non_zero_elements = abs(presentation[presentation != 0])
    non_zero_elements.sort() 
    is_trivial = np.array_equal(non_zero_elements, np.arange(1, 3))
    return is_trivial


# Returns the set of trivial states of the right length (i.e. 8 states )
def generate_trivial_states(max_relator_length):
    """ 
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


# Replace the i'th relation r_i by r_ir_j^{sign}.
def concat(rels, nrel, i, j, sign, lengths):
    """
    
    
    """
    rels = rels.copy()
    rel1 = rels[i * nrel : (i + 1) * nrel]

    lengths = lengths.copy()

    if sign == 1:
        rel2 = rels[j * nrel : (j + 1) * nrel]
    elif j:
        rel2 = -rels[(j + 1) * nrel - 1 : j * nrel - 1 : -1]
    else:
        rel2 = -rels[nrel - 1 :: -1]

    rel1 = rel1[rel1.nonzero()]
    rel2 = rel2[rel2.nonzero()]

    len1 = len(rel1)
    len2 = len(rel2)

    acc = 0
    while acc < min(len1, len2) and rel1[-1 - acc] == -rel2[acc]:
        acc += 1

    new_size = len1 + len2 - 2 * acc

    if new_size <= nrel:
        lengths[i] = new_size
        rels[i * nrel : i * nrel + len1 - acc] = rel1[: len1 - acc]
        rels[i * nrel + len1 - acc : i * nrel + new_size] = rel2[acc:]
        rels[i * nrel + new_size : (i + 1) * nrel] = 0

    return rels, lengths


# Conjugate the i'th relation by the j'th generator (i is 0 -- ngen-1, j is 1 -- ngen, sign = +/- 1 denoting conjugation by j or -j)
def conjugate(rels, nrel, i, j, sign, lengths):
    rels = rels.copy()
    rel = rels[i * nrel : (i + 1) * nrel]
    rel = rel[rel.nonzero()]

    lengths = lengths.copy()

    start_sign = sign * j
    end_sign = -sign * j
    rel_size = len(rel)

    if rel[-1] == start_sign:
        end_cancel = 1
    else:
        end_cancel = 0

    if rel[0] == end_sign:
        start_cancel = 1
    else:
        start_cancel = 0

    new_size = rel_size + 2 - 2 * (start_cancel + end_cancel)

    if new_size <= nrel:
        lengths[i] = new_size

        rels[
            i * nrel
            + 1
            - start_cancel : i * nrel
            + 1
            + rel_size
            - 2 * start_cancel
            - end_cancel
        ] = rel[start_cancel : rel_size - end_cancel]

        if not start_cancel:
            rels[i * nrel] = start_sign

        if not end_cancel:
            rels[i * nrel + rel_size + 1 - 2 * start_cancel] = end_sign

        if start_cancel and end_cancel:
            rels[i * nrel + new_size : i * nrel + new_size + 2] = 0

    return rels, lengths

# Defines AC moves..
# we encode swap, we will mostly avoid it.
# 1. r_1 --> r_1 r_0
# 2. r_0 --> r_0 r_1^{-1}
# 3. r_1 --> r_1 r_0^{-1}
# 4. r_0 --> r_0 r_1
# 5: r_1 --> x_0^{-1} r_1 x_0
# 6: r_0 ---> x_1^{-1} r_0 x_1
# 7: r_1 --> x_1^{-1} r_1 x_1
# 8: r_0 ---> x_0 r_0 x_0^{-1}
# 9: r_1 --> x_0 r_1 x_0^{-1}
# 10: r_0 --> x_1 r_0 x_1^{-1}
# 11: r_1 --> x_1 r_1 x_1^{-1}
# 12: r_0 --> x_0^{-1} r_0 x_0
# odd n affect r_1, even n affect r_0
# 1-4 concatenate, 5-12 conjugate
def ACMove(n, rels, ngen, nrel, lengths, full=True):

    # if n == 0:
    #  return full_simplify(rels, ngen, nrel)

    if n in [1, 2, 3, 4]:
        i = n % 2  # i = 0 for n even, 1 for n odd
        sign = ((n - i) // 2) % 2
        rels, lengths = concat(
            rels, nrel, i, 1 - i, (-1) ** sign, lengths
        )  # TODO: concat takes lengths
        return full_simplify(
            rels, ngen, nrel, lengths, full
        )  # TODO: full_simplify returns rels, lengths

    elif n in [5, 6, 7, 8, 9, 10, 11, 12]:
        i = n % 2  # i = 0 for even, 1 for odd
        j = ((n - i) // 2) % 2
        sign = ((n - i - 2 * j) // 4) % 2
        rels, lengths = conjugate(
            rels, nrel, i, j + 1, (-1) ** sign, lengths
        )  # TODO: concat takes lengths
        return full_simplify(
            rels, ngen, nrel, lengths, full
        )  # TODO: full_simplify returns rels, lengths

    print("Error AC move not found")


class ACEnv(Env):
    def __init__(self, config):

        self.n = config["n_gen"]
        self.max_length = config["max_length"]
        self.state = config["init_presentation"]
        self.initial_state = np.copy(config["init_presentation"])
        self.max_count_steps = config["max_count_steps"]
        self.count_steps = 0
        self.lengths = [
            len_words(self.state[i * self.max_length : (i + 1) * self.max_length])
            for i in range(self.n)
        ]
        self.max_reward = self.max_count_steps * self.max_length * self.n
        # self.min_lengths = self.lengths.copy()
        # self.min_total_length = sum(self.lengths)
        self.actions = []

        self.inverse_actions = {
            1: 3,
            2: 4,
            5: 9,
            6: 10,
            7: 11,
            8: 12,
            3: 1,
            4: 2,
            9: 5,
            10: 6,
            11: 7,
            12: 8,
        }

        if config["use_supermoves"]:
            self.supermoves = {
                13: (2, 9, 4, 1, 1),  # length = 5, appears 121 times
                14: (3, 5, 11, 3, 9, 11, 5, 1),  # length = 8, appears 26 times
                15: (4, 12, 2, 4, 4, 10, 3, 2, 9, 4),  # length = 10, apppears 19 times
                16: (
                    2,
                    12,
                    2,
                    10,
                    10,
                    12,
                    4,
                    12,
                    2,
                    4,
                    4,
                    10,
                    3,
                    2,
                    9,
                ),  # length = 15, appears 15 times
            }

            n_supermoves = len(self.supermoves)
            original_items = list(self.supermoves.items())

            for key, val in original_items:
                self.supermoves[key + n_supermoves] = tuple(
                    self.inverse_actions[a] for a in reversed(val)
                )

            self.action_space = Discrete(12 + len(self.supermoves))

        else:
            self.supermoves = None
            self.action_space = Discrete(12)

        low = np.ones(self.max_length * self.n, dtype=np.int8) * (-self.n)
        high = np.ones(self.max_length * self.n, dtype=np.int8) * (self.n)
        self.observation_space = Box(low, high, dtype=np.int8)

        if len(self.state) != 2 * self.max_length:
            print("There is an issue with length of relators.")

    def step(self, action):
        # action is in [0,11] but the input to ACMove is in [1,12] so we give action+1 as input to ACMove.
        self.actions.append(int(action + 1))
        # if action + 1 is a supermove, apply all actions in the supermove
        if self.supermoves and action + 1 in self.supermoves.keys():
            for a in self.supermoves[action + 1]:
                self.state, self.lengths = ACMove(
                    a, self.state, self.n, self.max_length, self.lengths
                )
        else:
            self.state, self.lengths = ACMove(
                action + 1, self.state, self.n, self.max_length, self.lengths
            )

        # self.min_lengths = [min(self.lengths[i], self.min_lengths[i]) for i in range(self.n)]
        # self.min_total_length = min(self.min_total_length, sum(self.lengths))

        done = sum(self.lengths) == 2
        reward = self.max_reward * done - sum(self.lengths) * (1 - done)

        self.count_steps += 1
        truncated = self.count_steps >= self.max_count_steps

        # record min_lengths before calling reset.
        # info = {'min_lengths': self.min_lengths, 'min_total_length': self.min_total_length}

        # if done:
        #   info['actions'] = self.actions.copy()

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
            len_words(self.state[i * self.max_length : (i + 1) * self.max_length])
            for i in range(self.n)
        ]
        self.count_steps = 0
        self.actions = []
        # self.min_lengths = self.lengths.copy()
        # self.min_total_length = sum(self.lengths)
        return self.state, {}

    def render(self):
        pass

    print("AC ENV LOADED")