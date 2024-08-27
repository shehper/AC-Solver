"""
Implements Andrews-Curtis (AC) Environment for balanced presentations with two generators.
"""

from dataclasses import dataclass, field
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from ac_solver.envs.ac_moves import ACMove


@dataclass
class ACEnvConfig:
    max_relator_length: int = 7
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array(
            [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0]
        )
    )
    horizon_length: int = 1000
    use_supermoves: bool = False
    max_count_steps = horizon_length  # Alias for backwards compatibility

    # assert (
    #     len(self.state) == self.n_gen * self.max_relator_length
    # ), f"The total length of initial_state = {len(config.initial_state)} must be equal \
    #         to  {2 * self.max_relator_length}."  TODO: fix this

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            max_relator_length=config_dict.get(
                "max_relator_length", cls().max_relator_length
            ),
            initial_state=np.array(
                config_dict.get("initial_state", cls().initial_state)
            ),
            horizon_length=config_dict.get("horizon_length", cls().horizon_length),
            use_supermoves=config_dict.get("use_supermoves", cls().use_supermoves),
        )


class ACEnv(Env):
    def __init__(self, config: ACEnvConfig = ACEnvConfig()):
        self.n_gen = 2
        self.max_relator_length = config.max_relator_length
        # 3 names for the same thing: state, initial_state, initial_state. Must consolidate.
        self.state = config.initial_state
        self.initial_state = np.copy(config.initial_state)
        self.horizon_length = config.horizon_length  # call it horizon_length
        if config.use_supermoves:
            raise NotImplementedError(
                "ACEnv with supermoves is not yet implemented in this library."
            )

        self.count_steps = 0
        self.lengths = [
            np.count_nonzero(
                self.state[
                    i * self.max_relator_length : (i + 1) * self.max_relator_length
                ]
            )
            for i in range(self.n_gen)
        ]

        low = np.ones(self.max_relator_length * self.n_gen, dtype=np.int8) * (
            -self.n_gen
        )
        high = np.ones(self.max_relator_length * self.n_gen, dtype=np.int8) * (
            self.n_gen
        )
        self.observation_space = Box(low, high, dtype=np.int8)
        self.action_space = Discrete(12)
        self.max_reward = self.horizon_length * self.max_relator_length * self.n_gen
        self.actions = []

    def step(self, action):
        self.actions += [action]
        self.state, self.lengths = ACMove(
            action, self.state, self.max_relator_length, self.lengths
        )

        done = sum(self.lengths) == 2
        reward = self.max_reward * done - sum(self.lengths) * (1 - done)

        self.count_steps += 1
        truncated = self.count_steps >= self.horizon_length

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
