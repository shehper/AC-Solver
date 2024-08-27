"""
`parse_args` function contains all of the arguments that a user may give through command line.
"""

import argparse
from distutils.util import strtobool
from os.path import basename


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--wandb-log",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="AC-Solver-PPO",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )

    # Environment specific arguments
    parser.add_argument(
        "--fixed-init-state",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="""each rollout may either start from the same fixed state or from one of many possible states. 
            If False (default), I use a files containing presentations of Miller-Schupp series. 
            Which file is chosen is determined by states-type arg (see below).
            If True, the presentation is specified by relator1, relator2 and max-length.""",
    )
    parser.add_argument(
        "--states-type",
        type=str,
        default="all",
        help="The type of states that must be loaded. Can take values solved, unsolved, or all.",
    )
    parser.add_argument(
        "--repeat-solved-prob",
        type=float,
        default=0.25,
        help="Probability of choosing an already solved state after all states have been attempted at least once.",
    )
    parser.add_argument(
        "--max-relator-length",
        type=int,
        default=7,
        help="the maximum length a relator is allowed to take when acted by AC moves",
    )
    parser.add_argument(
        "--relator1",
        nargs="+",
        type=int,
        default=[1, 1, -2, -2, -2],
        help="first relator of the initial presentation (default: AK(2)).",
    )
    parser.add_argument(
        "--relator2",
        nargs="+",
        type=int,
        default=[1, 2, 1, -2, -1, -2],
        help="second relator of the initial presentation (default: AK(2)).",
    )
    parser.add_argument(
        "--horizon-length",
        # in previous runs, this used to be called `max-env-steps`.
        type=int,
        default=2000,
        help="number of environment steps after which a rollout is truncated.",
    )
    parser.add_argument(
        "--use_supermoves",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to use supermoves or not; default: False",
    )

    # Architecture specific arguments
    # TODO: Include a shared layer vs not shared layer argument
    parser.add_argument(
        "--nodes-counts",
        nargs="+",
        type=int,
        default=[256, 256],
        help="list of length=number of hidden layers. ith element is the number of nodes \
            in the ith hidden layer.",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--is-loss-clip",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="loss is clip loss if True else KL-penalty loss",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="coefficient of KL-penalty loss; used if is-loss-clip=False",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--warmup-period",
        type=float,
        default=0.0,
        help="the ratio of total-timesteps over which learning rate should be increased linearly starting from 0",
    )
    parser.add_argument(
        "--lr-decay",
        type=str,
        default="linear",
        help="how to decay lr. should be either linear or cosine.",
    )
    parser.add_argument(
        "--min-lr-frac",
        type=float,
        default=0.0,
        help="fraction of maximum learning rate to which to anneal it.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="the number of parallel game environments.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs", type=int, default=1, help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--norm-rewards",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles reward normalization",
    )
    parser.add_argument(
        "--clip-rewards",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles reward clipping to [-10, 1000] by default. Can change bounds through min_rew and max_rew",
    )
    parser.add_argument(
        "--min-rew",
        type=int,
        default=-10,
        help="If clipping rewards, this is the lower bound",
    )
    parser.add_argument(
        "--max-rew",
        type=int,
        default=1000,
        help="If clipping rewards, this is the upper bound",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00001,
        help="epsilon hyperparameter for PyTorch Adam Optimizer",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    assert (
        0.0 <= args.warmup_period <= 1.0
    ), "warmup period should be less than 1.0 as it is the fraction of total timesteps"
    assert args.lr_decay in [
        "linear",
        "cosine",
    ], f"lr-decay must be linear or cosine, not {args.lr_decay}. Other LR schedules not supported yet"
    assert (
        0.0 <= args.min_lr_frac <= 1.0
    ), "min-lr-frac is the fraction of maximum lr to which we anneal."
    return args
