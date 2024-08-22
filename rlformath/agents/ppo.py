"""
Implementation of PPO for AC graph.

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

# try:
#   sys.path.insert(2, head_dir+'/gpt_data_and_model/gpt')
#   from model import GPTConfig, GPT
# except: 
#    print("gpt_data_and_model folder must lie in the second parent directory of this file, \
#          it must contain gpt/model.py that defines a GPT model.")
   


def to_array(rel1, rel2, max_length):
    assert 0 not in rel1 and 0 not in rel2, "rel1 and rel2 must not be padded with zeros."
    assert max_length >= max(len(rel1), len(rel2)), "max_length must be > max of lengths of rel1, rel2"

    return  np.array(rel1 + [0]*(max_length-len(rel1)) + rel2 + [0]*(max_length-len(rel2)), 
                             dtype=np.int8)

def to_list_of_lists(pres):
    """pres is a list padded with zeros, e.g. [1, 0, ...., 0, 2, 0, ...., 0] for trivial presentation."""
    """returns the pres in the form of a list of lists, e.g. [[1], [2]]"""
    assert type(pres) == list, "pres must be a list"
    len_arr = len(pres) # obtain length of the list
     # obtain length of each relator
    len1, len2 = np.count_nonzero(np.array(pres[:len_arr//2])), np.count_nonzero(np.array(pres[len_arr//2:]))
    # extract each relator
    rel1, rel2 = pres[:len1], pres[len_arr//2:len_arr//2+len2]
    rels = [rel1, rel2]
    return rels

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_net(nodes_counts, stdev):
    """
    # TODO: perhaps should be called get_hidden_layers
    """
    layers = [ layer_init(nn.Linear(nodes_counts[0], nodes_counts[1])), nn.Tanh()]
    for i in range(1, len(nodes_counts)-2): # layernum = 3
        layers.append(layer_init(nn.Linear(nodes_counts[i], nodes_counts[i+1])))
        layers.append(nn.Tanh())
    layers.append(layer_init(nn.Linear(nodes_counts[-2], nodes_counts[-1]), std=stdev))
    return layers

class Agent(nn.Module):
    def __init__(self, envs, nodes_counts, gptconf, use_transformer=True):
        super(Agent, self).__init__()
        self.use_transformer = use_transformer
        # As it is, we are using shared layers with self.use_transformer and unshared layers without.
        # That is why, the code below may seem confusing. I might fix that at a later time.
        if self.use_transformer:
            raise NotImplementedError("use-transformer=True requires importing transformer from nanoGPT") # TODO
            # gptconf['block_size'] = np.array(envs.single_observation_space.shape).prod()
            # gptconf = GPTConfig(**gptconf)
            # self.network = GPT(gptconf)
            # self.critic = layer_init(nn.Linear(6, 1), std=0.01)
            # self.actor = layer_init(nn.Linear(6, envs.single_action_space.n), std=1)
        else:
            # nodes_counts: list of number of nodes in each layer that is not input or output layer
            self.critic_nodes = [np.array(envs.single_observation_space.shape).prod()] + nodes_counts + [1]
            self.actor_nodes = [np.array(envs.single_observation_space.shape).prod()] + nodes_counts + [envs.single_action_space.n]
            self.critic = nn.Sequential(*get_net(self.critic_nodes, 1.0))
            self.actor = nn.Sequential(*get_net(self.actor_nodes, 0.01))

    def get_value(self, x):
        if self.use_transformer:
            raise NotImplementedError("use-transformer=True requires importing transformer from nanoGPT") # TODO
            # # TODO: Make this tokenizer step more efficient based on the comment below?
            # x = torch.where(x < 0, 2-x, x).to(dtype=torch.int64)
            # return self.critic(self.network(x)[0][:, -1, :]) 
        else:
            return self.critic(x) 

    def get_action_and_value(self, x, action=None):
        if self.use_transformer:
            x = torch.where(x < 0, 2-x, x).to(dtype=torch.int64)
            hidden = self.network(x)[0][:, -1, :]
            logits = self.actor(hidden)
            value = self.critic(hidden)
        else:
            logits = self.actor(x)
            value = self.critic(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
    
def make_env(presentation, args):
    def thunk():
        env_config = {'n_gen': 2, 'init_presentation': presentation, 'max_length': args.max_length, 
                        'max_count_steps': args.max_env_steps, 'use_supermoves': args.use_supermoves}
        env = ACEnv(env_config)
        if args.norm_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        if args.clip_rewards:
            assert args.min_rew < args.max_rew, "min-rew must be less than max-rew"
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, args.min_rew, args.max_rew))
        return env 
    return thunk

def get_pres(initial_states, pres_number, max_length): 
    rel1, rel2 = to_list_of_lists(initial_states[pres_number])
    return to_array(rel1, rel2, max_length) 
    
def get_initial_states(states_type):
    # load initial presentations from a file. they are already sorted by n, i.e. by their hardness
    assert states_type in ["solved", "all"], "states-type must be solved, or all"
    try:
        with resources.open_text('rlformath.search.miller_schupp.data', f'{states_type}_miller_schupp_presentations.txt') as file:
            initial_states = [literal_eval(line[:-1]) for line in file]
            print(f"loaded {len(initial_states)} presentations from {states_type}_miller_schupp_presentations.txt.")
        return initial_states
    except:
        return None
    
def get_curr_lr(n_update, 
                lr_decay,
                warmup,
                max_lr,
                min_lr,
                total_updates):

    # in the training loop, updates are 1-indexed. In this code, they are 0-indexed. 
    n_update = n_update - 1
    total_updates = total_updates - 1
    
    warmup_period_end = total_updates * warmup

    if warmup_period_end > 0 and n_update <= warmup_period_end:
        lrnow = max_lr * n_update / warmup_period_end
    else:
        if lr_decay == "linear":
            slope = (max_lr - min_lr)/(warmup_period_end - total_updates)
            intercept = max_lr - slope * warmup_period_end
            lrnow = slope * n_update + intercept

        elif lr_decay == "cosine":
            cosine_arg = (n_update - warmup_period_end) / (total_updates - warmup_period_end) * math.pi
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplemented("only linear and cosine lr-schedules are available")    
    return lrnow  

