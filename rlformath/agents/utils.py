"""
This file contains various helper functions for agents. 

"""
import math
import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn
import gymnasium as gym
from importlib import resources
from ast import literal_eval
from rlformath.envs.ac_env import ACEnv

def convert_relators_to_presentation(relator1, relator2, max_relator_length):
    """
    Converts two lists representing relators into a single numpy array, padding each relator with zeros
    to match the specified maximum length.

    Parameters:
    relator1 (list of int): The first relator, must not contain zeros.
    rel2 (list of int): The second relator, must not contain zeros.
    max_relator_length (int): The maximum allowed length for each relator.

    Returns:
    np.ndarray: A numpy array of dtype int8, containing the two relators concatenated and zero-padded to max_length.
    """
    
    # Ensure relators do not contain zeros and max_relator_length is sufficient
    assert 0 not in relator1 and 0 not in relator2, "relator1 and relator2 must not be padded with zeros."
    assert max_relator_length >= max(len(relator1), len(relator2)), "max_relator_length must be greater than or equal to the lengths of relator1 and rel2."
    assert isinstance(relator1, list) and isinstance(relator2, list), f"got types {type(relator1)} for relator1 and {type(relator2)} for relator2"
     
    padded_relator1 = relator1 + [0] * (max_relator_length - len(relator1))
    padded_relator2 = relator2 + [0] * (max_relator_length - len(relator2))

    return np.array(padded_relator1 + padded_relator2, dtype=np.int8)

def change_max_relator_length_of_presentation(presentation, new_max_length):

    old_max_length = len(presentation) // 2

    first_word_length = np.count_nonzero(presentation[:old_max_length])
    second_word_length = np.count_nonzero(presentation[old_max_length:])

    relator1 = presentation[:first_word_length]
    relator2 = presentation[old_max_length:old_max_length+second_word_length]

    new_presentation = convert_relators_to_presentation(relator1=relator1, relator2=relator2, max_relator_length=new_max_length)
    return new_presentation


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
    
    layers.append(initialize_layer(nn.Linear(nodes_counts[-2], nodes_counts[-1]), std=std))
    
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
    
def make_env(presentation, args):
    """
    Creates an environment initialization function (thunk) with the specified configuration.

    Parameters:
    presentation (list): The initial presentation configuration for the environment.
    args (Namespace): A set of arguments containing the environment parameters such as max_relator_length, max_env_steps, 
                      use_supermoves, norm_rewards, gamma, clip_rewards, min_rew, and max_rew.

    Returns:
    function: A thunk (a function with no arguments) that initializes and returns the environment when called.
    """
    

    def thunk():

        env_config = {
            'n_gen': 2,
            'init_presentation': presentation,
            'max_relator_length': args.max_relator_length,
            'max_count_steps': args.max_env_steps,
            'use_supermoves': args.use_supermoves
        }

        env = ACEnv(env_config)

        # optionally normalize and / or clip rewards
        if args.norm_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        
        if args.clip_rewards:
            assert args.min_rew < args.max_rew, "min_rew must be less than max_rew"
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, args.min_rew, args.max_rew))
        
        return env 

    return thunk
    
def load_initial_states_from_text_file(states_type):
    """
    Loads initial presentations from a text file based on the specified state type. The presentations
    are sorted by their hardness.

    Parameters:
    states_type (str): The type of states to load. Must be either "solved" or "all".

    Returns:
    list: A list of presentations loaded from the text file.

    Raises:
    AssertionError: If states_type is not "solved" or "all".
    """
    assert states_type in ["solved", "all"], "states_type must be 'solved' or 'all'"

    file_name = f'{states_type}_miller_schupp_presentations.txt'
    with resources.open_text('rlformath.search.miller_schupp.data', file_name) as file:
        initial_states = [literal_eval(line.strip()) for line in file]

    print(f"Loaded {len(initial_states)} presentations from {file_name}.")
    return initial_states

def get_curr_lr(n_update, lr_decay, warmup, max_lr, min_lr, total_updates):
    """
    Calculates the current learning rate based on the update step, learning rate decay schedule, 
    warmup period, and other parameters.

    Parameters:
    n_update (int): The current update step (1-indexed).
    lr_decay (str): The type of learning rate decay to apply ("linear" or "cosine").
    warmup (float): The fraction of total updates to be used for the learning rate warmup.
    max_lr (float): The maximum learning rate.
    min_lr (float): The minimum learning rate.
    total_updates (int): The total number of updates.

    Returns:
    float: The current learning rate.

    Raises:
    NotImplementedError: If an unsupported lr_decay type is provided.
    """
    # Convert to 0-indexed for internal calculations
    n_update -= 1
    total_updates -= 1
    
    # Calculate the end of the warmup period
    warmup_period_end = total_updates * warmup

    if warmup_period_end > 0 and n_update <= warmup_period_end:
        lrnow = max_lr * n_update / warmup_period_end
    else:
        if lr_decay == "linear":
            slope = (max_lr - min_lr) / (warmup_period_end - total_updates)
            intercept = max_lr - slope * warmup_period_end
            lrnow = slope * n_update + intercept

        elif lr_decay == "cosine":
            cosine_arg = (n_update - warmup_period_end) / (total_updates - warmup_period_end) * math.pi
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplementedError("Only 'linear' and 'cosine' lr-schedules are available.")
    
    return lrnow

