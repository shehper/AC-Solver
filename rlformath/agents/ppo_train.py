"""
Implementation of PPO for AC graph.

"""
import math
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.optim import Adam
import gymnasium as gym
import argparse
import random
import time, uuid
import wandb
from distutils.util import strtobool
from collections import deque
from multiprocessing import cpu_count
from os import makedirs
from os.path import dirname, abspath, basename, join
from rlformath.agents.ppo import to_array, Agent, make_env, \
      get_pres, get_curr_lr, load_initial_states_from_text_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-log", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="PPO-AC",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    # Environment specific arguments
    parser.add_argument("--fixed-init-state", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="""each rollout may either start from the same fixed state or from one of many possible states. 
            If False (default), I use a files containing presentations of Miller-Schupp series. 
            Which file is chosen is determined by states-type arg (see below).
            If True, the presentation is specified by relator1, relator2 and max-length.""")
    parser.add_argument("--states-type", type=str, default="all", 
        help="The type of states that must be loaded. Can take values solved, unsolved, or all.")
    parser.add_argument("--repeat-solved-prob", type=float, default=0.25, 
        help="Probability of choosing an already solved state after all states have been attempted at least once.")
    parser.add_argument("--max-length", type=int, default=7,
        help="the maximum length a relator is allowed to take when acted by AC moves")
    parser.add_argument('--relator1', nargs='+', type=int, default=[1, 1, -2, -2, -2], 
        help="first relator of the initial presentation (default: AK(2)).")
    parser.add_argument('--relator2', nargs='+', type=int, default=[1,2,1,-2,-1,-2], 
        help="second relator of the initial presentation (default: AK(2)).") 
    parser.add_argument("--max-env-steps", type=int, default=2000,
        help="number of environment steps after which a rollout is truncated.")
    parser.add_argument("--use_supermoves", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use supermoves or not; default: False")
    
    # Architecture specific arguments
    # TODO: Include a shared layer vs not shared layer argument
    parser.add_argument("--nodes-counts", nargs='+', type=int, default=[256, 256],
        help="list of length=number of hidden layers. ith element is the number of nodes \
            in the ith hidden layer.")
    
    # Algorithm specific arguments
    parser.add_argument('--is-loss-clip', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help='loss is clip loss if True else KL-penalty loss')
    parser.add_argument('--beta', type=float, default=0.9, 
        help='coefficient of KL-penalty loss; used if is-loss-clip=False')
    parser.add_argument("--total-timesteps", type=int, default=int(1.1e9),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--warmup-period", type=float, default=0.0,
        help="the ratio of total-timesteps over which learning rate should be increased linearly starting from 0")
    parser.add_argument("--lr-decay", type=str, default="linear",
        help="how to decay lr. should be either linear or cosine.")
    parser.add_argument("--min-lr-frac", type=float, default=0.0,
        help="fraction of maximum learning rate to which to anneal it.")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments.")
    parser.add_argument("--num-steps", type=int, default=2000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--norm-rewards", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles reward normalization")
    parser.add_argument("--clip-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles reward clipping to [-10, 1000] by default. Can change bounds through min_rew and max_rew")
    parser.add_argument("--min-rew", type=int, default=-10, 
        help="If clipping rewards, this is the lower bound")    
    parser.add_argument("--max-rew", type=int, default=1000, 
        help="If clipping rewards, this is the upper bound")   
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.01,
        help="the target KL divergence threshold")
    parser.add_argument("--epsilon", type=float, default=0.00001,
        help="epsilon hyperparameter for PyTorch Adam Optimizer")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    assert 0.0 <= args.warmup_period <= 1.0, "warmup period should be less than 1.0 as it is the fraction of total timesteps"
    assert args.lr_decay in ["linear", "cosine"], f"lr-decay must be linear or cosine, not {args.lr_decay}. Other LR schedules not supported yet"
    assert 0.0 <= args.min_lr_frac <= 1.0, "min-lr-frac is the fraction of maximum lr to which we anneal."
    return args

if __name__ == '__main__':

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # if initial states are to be loaded from a file, do that.
    if not args.fixed_init_state:
        initial_states = load_initial_states_from_text_file(states_type=args.states_type)
        
    # initiate envs
    if args.fixed_init_state: 
        envs = gym.vector.SyncVectorEnv([make_env(to_array(args.relator1, args.relator2, args.max_length), args) 
                                         for _ in range(args.num_envs)])
    else:
        args.max_length = 36 # update maxlength to max(max(18, 4n+2)) for 1 <= n <= 7 
        assert args.num_envs <= len(initial_states), "modify initiation of envs if num_envs > number of initial states"
        envs = gym.vector.SyncVectorEnv([make_env(get_pres(initial_states, i, args.max_length), args) for i in range(args.num_envs)])
        # keep a record of current initial states and all solved/unsolved states
        curr_states = list(range(args.num_envs)) # states being processed at the moment
        states_processed = set(curr_states) # set of states either under process or have been processed
        success_record = {'solved': set(), 'unsolved': set(range(len(initial_states))) }
        ACMoves_hist = {}

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    agent = Agent(envs, args.nodes_counts).to(device)
    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=args.epsilon)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0]).to(device) # get first observation
    next_done = torch.zeros(args.num_envs).to(device) # get first done
    num_updates = args.total_timesteps // args.batch_size
    episodic_return = np.array([0]*args.num_envs)
    episodic_length = np.array([0]*args.num_envs)
    episode = 0
    returns_queue = deque([0], maxlen=100)
    lengths_queue = deque([0], maxlen=100)
    round1_complete = False # whether we have already chosen each element of initial_states at least once to initiate rollout
    beta = None if args.is_loss_clip else args.beta

    run_name = f"{args.exp_name}_ppo-ffn-nodes_{args.nodes_counts}_{uuid.uuid4()}"
    out_dir = f"out/{run_name}"
    makedirs(out_dir, exist_ok=True)
    if args.wandb_log:
        run = wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config = vars(args),
            save_code=True,
        )

    start_time = time.time()
    print(f'total number of timesteps: {args.total_timesteps}, updates: {num_updates}')
    for update in range(1, num_updates + 1):
        print(f'Now collecting data for update number {update}')

        # using different seed for each update to ensure reproducibility of paused-and-resumed runs
        random.seed(args.seed + update)
        np.random.seed(args.seed + update)
        torch.manual_seed(args.seed + update)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            lrnow = get_curr_lr(n_update=update, 
                                lr_decay=args.lr_decay,
                                warmup=args.warmup_period,
                                max_lr=args.learning_rate,
                                min_lr=args.learning_rate * args.min_lr_frac,
                                total_updates=num_updates)
            optimizer.param_groups[0]["lr"] = lrnow

        # collecting and recording data
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs 
            dones[step] = next_done # contains 1 if done else 0

            start = time.time()
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs) # shapes: n_envs, n_envs, n_envs, (n_envs, 1)
                values[step] = value.flatten() # num_envs
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, infos = envs.step(action.cpu().numpy()) # step is taken on cpu
            rewards[step] = torch.tensor(reward).to(device).view(-1) # r_0 is the reward from taking a_0 in s_0
            episodic_return = episodic_return + reward
            episodic_length = episodic_length + 1
            
            _record_info = np.array([True if done[i] or truncated[i] else False for i in range(args.num_envs)])
            if _record_info.any():
                for i, el in enumerate(_record_info):
                    
                    if done[i]:
                        # if done, add curr_states[i] to 'solved' cases
                        if curr_states[i] in success_record['unsolved']:
                            success_record['unsolved'].remove(curr_states[i])
                            success_record['solved'].add(curr_states[i])
                        
                        # also if done, record the sequence of actions in ACMoves_hist
                        if curr_states[i] not in ACMoves_hist:
                            ACMoves_hist[curr_states[i]] = infos['final_info'][i]['actions']
                        else:
                            prev_path_length = len(ACMoves_hist[curr_states[i]])
                            new_path_length = len(infos['final_info'][i]['actions'])
                            if new_path_length < prev_path_length:
                                print(f"For state {curr_states[i]}, found a shorter path of length {new_path_length};\
                                 previous path length: {prev_path_length}.")
                                ACMoves_hist[curr_states[i]] = infos['final_info'][i]['actions']

                    # record+reset episode data, reset ith initial state to the next state in init_states                  
                    if el: 
                        # record and reset episode data
                        returns_queue.append(episodic_return[i])
                        lengths_queue.append(episodic_length[i])
                        #print(f"global_step={global_step}, episodic_return={episodic_return[i]}")
                        episode += 1
                        episodic_return[i], episodic_length[i] = 0, 0

                        # update next_obs to have the next initial state
                        prev_state = curr_states[i]
                        round1_complete = True if round1_complete or (max(states_processed) == len(initial_states) - 1) else False
                        if not round1_complete:
                            curr_states[i] = max(states_processed) + 1
                        else:
                            # TODO: If states-type=all, first choose from all solved presentations then choose from unsolved presentations
                            if len(success_record["solved"]) == 0 or \
                                (success_record['unsolved'] and random.uniform(0, 1) > args.repeat_solved_prob):
                                curr_states[i] = random.choice(list(success_record['unsolved'])) 
                            else:
                                curr_states[i] = random.choice(list(success_record['solved']))
                        states_processed.add(curr_states[i])
                        next_obs[i] = get_pres(initial_states, curr_states[i], args.max_length)
                        envs.envs[i].reset(options={'starting_state': next_obs[i]})
                        #print(f"Updating initial state of env {i} from {prev_state} to {curr_states[i]}")

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
        if not args.norm_rewards: # if not normalizing rewards through a NormalizeRewards Wrapper, rescale rewards manually.
            rewards /= envs.envs[0].max_reward
            normalized_returns = np.array(returns_queue) / envs.envs[0].max_reward
            normalized_lengths = np.array(lengths_queue) / args.max_env_steps
        else:
            normalized_returns = np.array(returns_queue)
            normalized_lengths = np.array(lengths_queue)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values # Where do we use returns?
            
        # flattening out the data collected from parallel environments
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) # num_envs * num_steps, obs_space.shape
        b_logprobs = logprobs.reshape(-1) # num_envs * num_steps
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value networks
        b_inds = np.arange(args.batch_size) # indices of batch_size
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size): 
                end = start + args.minibatch_size 
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) #.long() converts dtype to int64
                logratio = newlogprob - b_logprobs[mb_inds] 
                ratio = logratio.exp() # pi(a|s) / pi_old(a|s); is a tensor of 1s for epoch=0.

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                kl_var = (ratio - 1) - logratio # the random variable whose expectation gives approx kl
                with torch.no_grad():
                    approx_kl = kl_var.mean() # mean of (pi(a|s) / pi_old(a|s) - 1 - log(pi(a|s) / pi_old(a|s)))
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv: 
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                if args.is_loss_clip: # clip loss
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else: # KL-penalty loss
                    pg_loss2 = beta * kl_var
                    pg_loss = (pg_loss1 + pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1) # value computed by NN with updated parameters
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # can i implement this myself?
                optimizer.step()

            if args.is_loss_clip: # if clip loss and approx_kl > target kl, break
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            else: # if KL-penalty loss, update beta
                beta = beta / 2 if approx_kl < args.target_kl / 1.5 else (beta * 2 if approx_kl > args.target_kl * 1.5 else beta)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy() 
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.wandb_log:
            wandb.log({
                "charts/global_step": global_step,
                "charts/episode": episode,
                "charts/normalized_returns_mean": normalized_returns.mean(),
                "charts/normalized_lengths_mean": normalized_lengths.mean(),
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/solved": len(success_record['solved']),
                "charts/unsolved": len(success_record['unsolved']),
                "charts/highest_solved": max(success_record['solved']) if success_record['solved'] else -1,
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/explained_variance": explained_var,
                "losses/clipfrac": np.mean(clipfracs),
                "debug/advantages_mean": b_advantages.mean(), 
                "debug/advantages_std": b_advantages.std(),
            })

        if update > 0 and update % 100 == 0: # save a checkpoint every 100 updates
            checkpoint = {
                        'critic': agent.critic.state_dict(),
                        'actor': agent.actor.state_dict(),                        
                        'optimizer': optimizer.state_dict(),
                        'update': update,
                        'episode': episode,
                        'config': vars(args),
                        'mean_return': normalized_returns.mean(),
                        'success_record': success_record,
                        "value_loss": v_loss.item(),
                        "policy_loss": pg_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "approx_kl": approx_kl.item(),
                        "explained_var": explained_var,
                        "clipfrac": np.mean(clipfracs),
                        "global_step": global_step,
                        "round1_complete": round1_complete, 
                        "curr_states": curr_states, 
                        "states_processed": states_processed,
                        "ACMoves_hist": ACMoves_hist,
                        "supermoves": envs.envs[0].supermoves, # dict of supermoves or None
                    }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, join(out_dir, 'ckpt.pt'))

    envs.close()