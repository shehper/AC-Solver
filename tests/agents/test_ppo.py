import pytest
from unittest.mock import patch
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from argparse import Namespace
from ac_solver.envs.ac_env import ACEnv
from ac_solver.agents.args import parse_args
from ac_solver.agents.ppo_agent import initialize_layer, build_network, Agent
from ac_solver.agents.training import get_curr_lr
from ac_solver.agents.environment import make_env
from ac_solver.envs.utils import convert_relators_to_presentation


# Sample input data for testing
@pytest.fixture
def sample_args():
    return Namespace(
        exp_name="test_exp",
        seed=1,
        torch_deterministic=True,
        cuda=False,
        wandb_log=False,
        fixed_init_state=False,
        states_type="all",
        repeat_solved_prob=0.25,
        max_relator_length=7,
        relator1=[1, 1, -2, -2, -2],
        relator2=[1, 2, 1, -2, -1, -2],
        max_env_steps=2000,
        use_supermoves=False,
        nodes_counts=[256, 256],
        is_loss_clip=True,
        beta=0.9,
        total_timesteps=1000,
        learning_rate=2.5e-4,
        warmup_period=0.1,
        lr_decay="linear",
        min_lr_frac=0.0,
        num_envs=4,
        override_num_envs=False,
        override_lr=False,
        num_steps=10,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=1,
        norm_adv=True,
        norm_rewards=False,
        clip_rewards=True,
        min_rew=-10,
        max_rew=1000,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_nodrm=0.5,
        target_kl=0.01,
        epsilon=0.00001,
        batch_size=40,
        minibatch_size=10,
    )


def test_parse_args():
    test_args = ["script_name", "--exp-name", "test_exp", "--seed", "42"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.exp_name == "test_exp"
        assert args.seed == 42
        assert args.torch_deterministic is True
        assert args.cuda is True
        assert args.wandb_log is False


# Test layer initialization function
def test_initialize_layer():
    layer = nn.Linear(4, 2)
    initialized_layer = initialize_layer(layer)
    assert initialized_layer.weight.shape == torch.Size([2, 4])
    assert initialized_layer.bias.shape == torch.Size([2])


# Test get_net function
def test_get_net():
    nodes_counts = [4, 8, 2]
    layers = build_network(nodes_counts, 1.0)
    assert len(layers) == 3
    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.Tanh)


# Test Agent class initialization
def test_agent_initialization(sample_args):
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                convert_relators_to_presentation(
                    sample_args.relator1,
                    sample_args.relator2,
                    sample_args.max_relator_length,
                ),
                sample_args,
            )
        ]
    )
    agent = Agent(envs, sample_args.nodes_counts)
    assert isinstance(agent, Agent)


# Test Agent methods get_value and get_action_and_value
def test_agent_methods(sample_args):
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                convert_relators_to_presentation(
                    sample_args.relator1,
                    sample_args.relator2,
                    sample_args.max_relator_length,
                ),
                sample_args,
            )
        ]
    )
    agent = Agent(envs, sample_args.nodes_counts)

    obs = torch.randn(
        (sample_args.num_steps, sample_args.num_envs)
        + envs.single_observation_space.shape
    )

    value = agent.get_value(obs[0])
    assert value.shape == torch.Size([sample_args.num_envs, 1])

    action, log_prob, entropy, value = agent.get_action_and_value(obs[0])
    assert action.shape == torch.Size([sample_args.num_envs])
    assert log_prob.shape == torch.Size([sample_args.num_envs])
    assert entropy.shape == torch.Size([sample_args.num_envs])
    assert value.shape == torch.Size([sample_args.num_envs, 1])


# Test get_curr_lr function
# TODO: including more cases here can be useful
@pytest.mark.parametrize(
    "lr_decay, warmup, n_update, expected_lr",
    [
        ("linear", 0.1, 1, 0.0),
        ("linear", 0.0, 1, 2.5e-04),
        ("cosine", 0.0, 159, 0.0),
        ("cosine", 0.1, 100, 9.36e-05),
    ],
)
def test_get_curr_lr(lr_decay, warmup, n_update, expected_lr, sample_args):
    max_lr = sample_args.learning_rate
    min_lr = max_lr * sample_args.min_lr_frac
    total_updates = sample_args.total_timesteps // sample_args.batch_size

    lrnow = get_curr_lr(n_update, lr_decay, warmup, max_lr, min_lr, total_updates)
    assert np.isclose(lrnow, expected_lr, atol=1e-4)


# Test make_env function
def test_make_env(sample_args):
    presentation = convert_relators_to_presentation(
        sample_args.relator1, sample_args.relator2, sample_args.max_relator_length
    )
    env_thunk = make_env(presentation, sample_args)
    env = env_thunk()
    if sample_args.clip_rewards:
        assert isinstance(env, gym.wrappers.TransformReward)
    elif sample_args.norm_rewards:
        assert isinstance(env, gym.wrappers.NormalizeReward)
    else:
        assert isinstance(
            env, ACEnv
        ), f"expect env to be of type ACEnv, got {type(env)}"
