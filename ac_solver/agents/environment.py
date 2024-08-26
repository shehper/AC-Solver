import gymnasium as gym

from ac_solver.agents.utils import (
    make_env,
    load_initial_states_from_text_file,
    convert_relators_to_presentation,
    change_max_relator_length_of_presentation,
)

## initialize environments
def get_env(args):
    if args.fixed_init_state:
        # Convert provided relators to a presentation if a fixed initial state is used
        presentation = convert_relators_to_presentation(
            args.relator1, args.relator2, args.max_relator_length
        )

        # Create a vectorized environment with the same presentation for all environments
        envs = gym.vector.SyncVectorEnv(
            [make_env(presentation, args) for _ in range(args.num_envs)]
        )
    else:
        # Load initial states from a text file based on the specified states type
        initial_states = load_initial_states_from_text_file(
            states_type=args.states_type
        )

        # Ensure the number of environments does not exceed the number of available initial states
        assert args.num_envs <= len(
            initial_states
        ), "Expect number of environments to be less than number of distinct initial states for now; \
                                                     relaxing this condition is easy. Just edit the definition of envs below."  # TODO

        # Set the maximum relator length; default is 36, which is derived from max(4n+2) for 1 <= n <= 7
        # This can be modified for experimentation
        args.max_relator_length = 36
        initial_states = [
            change_max_relator_length_of_presentation(
                initial_state, args.max_relator_length
            )
            for initial_state in initial_states
        ]

        # Create a vectorized environment, each initialized with a different presentation from initial_states
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(presentation=initial_states[i], args=args)
                for i in range(args.num_envs)
            ]
        )

        # Track the current states being processed and keep a record of processed states
        curr_states = list(range(args.num_envs))  # States currently being processed
        states_processed = set(
            curr_states
        )  # Set of states that have been or are being processed

        # Record successes and failures for states
        success_record = {
            "solved": set(),  # States that have been successfully solved
            "unsolved": set(
                range(len(initial_states))
            ),  # States that have not yet been solved
        }

        # History of actions/moves taken in the AC algorithm
        ACMoves_hist = {}
    
    return envs, initial_states, curr_states, success_record, ACMoves_hist, states_processed 