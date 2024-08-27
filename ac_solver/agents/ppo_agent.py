"""
This file contains Agent class that implements actor and critic networks.
"""

import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn


def initialize_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given layer.

    Parameters:
    layer (nn.Module): The neural network layer to initialize.
    std (float): The standard deviation for orthogonal initialization of weights. Default is sqrt(2).
    bias_const (float): The constant value to initialize the biases. Default is 0.0.

    Returns:
    nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_network(nodes_counts, std=0.01):
    """
    Constructs a neural network with fully connected layers and Tanh activations based on the specified node counts.

    Parameters:
    nodes_counts (list of int): A list where each element represents the number of nodes in a layer.
    std (float): The standard deviation for initializing the final layer's weights. Default is 0.01.

    Returns:
    list: A list of layers (including activation functions) representing the neural network.
    """
    layers = [initialize_layer(nn.Linear(nodes_counts[0], nodes_counts[1])), nn.Tanh()]

    for i in range(1, len(nodes_counts) - 2):
        layers.append(initialize_layer(nn.Linear(nodes_counts[i], nodes_counts[i + 1])))
        layers.append(nn.Tanh())

    layers.append(
        initialize_layer(nn.Linear(nodes_counts[-2], nodes_counts[-1]), std=std)
    )

    return layers


class Agent(nn.Module):
    """
    A reinforcement learning agent that includes both a critic network for value estimation
    and an actor network for policy generation.

    Attributes:
    critic (nn.Sequential): The neural network used for value estimation.
    actor (nn.Sequential): The neural network used for policy generation.
    """

    def __init__(self, envs, nodes_counts):
        """
        Initializes the Agent with specified environment and node counts for the neural networks.

        Parameters:
        envs (gym.Env): The environment for which the agent is being created.
        nodes_counts (list of int): A list where each element represents the number of nodes in a hidden layer.
        """
        super(Agent, self).__init__()

        input_dim = np.prod(envs.single_observation_space.shape)
        self.critic_nodes = [input_dim] + nodes_counts + [1]
        self.actor_nodes = [input_dim] + nodes_counts + [envs.single_action_space.n]

        self.critic = nn.Sequential(*build_network(self.critic_nodes, 1.0))
        self.actor = nn.Sequential(*build_network(self.actor_nodes, 0.01))

    def get_value(self, x):
        """
        Computes the value of a given state using the critic network.

        Parameters:
        x (torch.Tensor): The input tensor representing the state.

        Returns:
        torch.Tensor: The value of the given state.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Computes the action to take and its associated value, log probability, and entropy.

        Parameters:
        x (torch.Tensor): The input tensor representing the state.
        action (torch.Tensor, optional): The action to evaluate. If None, a new action will be sampled.

        Returns:
        tuple: A tuple containing the action, its log probability, the entropy of the action distribution, and the value of the state.
        """
        logits = self.actor(x)
        value = self.critic(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value
