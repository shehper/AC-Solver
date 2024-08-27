"""
Implementation of PPO for AC graph.

# TODO: include some examples here.

"""

import numpy as np
import torch
import random
from torch.optim import Adam
from ac_solver.agents.ppo_agent import Agent
from ac_solver.agents.args import parse_args
from ac_solver.agents.environment import get_env
from ac_solver.agents.training import ppo_training_loop


def train_ppo():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)

    agent = Agent(envs, args.nodes_counts).to(device)
    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=args.epsilon)

    ppo_training_loop(
        envs,
        args,
        device,
        optimizer,
        agent,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
    )

    envs.close()


if __name__ == "__main__":
    train_ppo()
