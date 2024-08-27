"""
ppo_training_loop function implements the training loop logic of PPO.
"""

import math
import random
import uuid
import wandb
from collections import deque
from tqdm import tqdm
from os import makedirs
from os.path import join
import numpy as np
import torch
from torch import nn


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
            cosine_arg = (
                (n_update - warmup_period_end)
                / (total_updates - warmup_period_end)
                * math.pi
            )
            lrnow = min_lr + (max_lr - min_lr) * (1 + math.cos(cosine_arg)) / 2

        else:
            raise NotImplementedError(
                "Only 'linear' and 'cosine' lr-schedules are available."
            )

    return lrnow


def ppo_training_loop(
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
):
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0]).to(device)  # get first observation
    next_done = torch.zeros(args.num_envs).to(device)  # get first done
    num_updates = args.total_timesteps // args.batch_size
    episodic_return = np.array([0] * args.num_envs)
    episodic_length = np.array([0] * args.num_envs)
    episode = 0
    returns_queue = deque([0], maxlen=100)
    lengths_queue = deque([0], maxlen=100)
    round1_complete = False  # whether we have already chosen each element of initial_states at least once to initiate rollout
    beta = None if args.is_loss_clip else args.beta

    run_name = f"{args.exp_name}_ppo-ffn-nodes_{args.nodes_counts}_{uuid.uuid4()}"
    out_dir = f"out/{run_name}"
    makedirs(out_dir, exist_ok=True)
    if args.wandb_log:
        run = wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    print(f"total number of timesteps: {args.total_timesteps}, updates: {num_updates}")
    for update in tqdm(
        range(1, num_updates + 1), desc="Training Progress", total=num_updates
    ):

        # using different seed for each update to ensure reproducibility of paused-and-resumed runs
        random.seed(args.seed + update)
        np.random.seed(args.seed + update)
        torch.manual_seed(args.seed + update)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            lrnow = get_curr_lr(
                n_update=update,
                lr_decay=args.lr_decay,
                warmup=args.warmup_period,
                max_lr=args.learning_rate,
                min_lr=args.learning_rate * args.min_lr_frac,
                total_updates=num_updates,
            )
            optimizer.param_groups[0]["lr"] = lrnow

        # collecting and recording data
        for step in tqdm(
            range(0, args.num_steps), desc=f"Rollout Phase - {update}", leave=False
        ):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done  # contains 1 if done else 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs
                )  # shapes: n_envs, n_envs, n_envs, (n_envs, 1)
                values[step] = value.flatten()  # num_envs
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, infos = envs.step(
                action.cpu().numpy()
            )  # step is taken on cpu
            rewards[step] = (
                torch.tensor(reward).to(device).view(-1)
            )  # r_0 is the reward from taking a_0 in s_0
            episodic_return = episodic_return + reward
            episodic_length = episodic_length + 1

            _record_info = np.array(
                [
                    True if done[i] or truncated[i] else False
                    for i in range(args.num_envs)
                ]
            )
            if _record_info.any():
                for i, el in enumerate(_record_info):

                    if done[i]:
                        # if done, add curr_states[i] to 'solved' cases
                        if curr_states[i] in success_record["unsolved"]:
                            success_record["unsolved"].remove(curr_states[i])
                            success_record["solved"].add(curr_states[i])

                        # also if done, record the sequence of actions in ACMoves_hist
                        if curr_states[i] not in ACMoves_hist:
                            ACMoves_hist[curr_states[i]] = infos["final_info"][i][
                                "actions"
                            ]
                        else:
                            prev_path_length = len(ACMoves_hist[curr_states[i]])
                            new_path_length = len(infos["final_info"][i]["actions"])
                            if new_path_length < prev_path_length:
                                ACMoves_hist[curr_states[i]] = infos["final_info"][i][
                                    "actions"
                                ]

                    # record+reset episode data, reset ith initial state to the next state in init_states
                    if el:
                        # record and reset episode data
                        returns_queue.append(episodic_return[i])
                        lengths_queue.append(episodic_length[i])
                        episode += 1
                        episodic_return[i], episodic_length[i] = 0, 0

                        # update next_obs to have the next initial state
                        prev_state = curr_states[i]
                        round1_complete = (
                            True
                            if round1_complete
                            or (max(states_processed) == len(initial_states) - 1)
                            else False
                        )
                        if not round1_complete:
                            curr_states[i] = max(states_processed) + 1
                        else:
                            # TODO: If states-type=all, first choose from all solved presentations then choose from unsolved presentations
                            if len(success_record["solved"]) == 0 or (
                                success_record["unsolved"]
                                and random.uniform(0, 1) > args.repeat_solved_prob
                            ):
                                curr_states[i] = random.choice(
                                    list(success_record["unsolved"])
                                )
                            else:
                                curr_states[i] = random.choice(
                                    list(success_record["solved"])
                                )
                        states_processed.add(curr_states[i])
                        next_obs[i] = initial_states[curr_states[i]]
                        envs.envs[i].reset(options={"starting_state": next_obs[i]})

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

        if (
            not args.norm_rewards
        ):  # if not normalizing rewards through a NormalizeRewards Wrapper, rescale rewards manually.
            rewards /= envs.envs[0].max_reward
            normalized_returns = np.array(returns_queue) / envs.envs[0].max_reward
            normalized_lengths = np.array(lengths_queue) / args.horizon_length
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
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values  # Where do we use returns?

        # flattening out the data collected from parallel environments
        b_obs = obs.reshape(
            (-1,) + envs.single_observation_space.shape
        )  # num_envs * num_steps, obs_space.shape
        b_logprobs = logprobs.reshape(-1)  # num_envs * num_steps
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value networks
        b_inds = np.arange(args.batch_size)  # indices of batch_size
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )  # .long() converts dtype to int64
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = (
                    logratio.exp()
                )  # pi(a|s) / pi_old(a|s); is a tensor of 1s for epoch=0.

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                kl_var = (
                    ratio - 1
                ) - logratio  # the random variable whose expectation gives approx kl
                with torch.no_grad():
                    approx_kl = (
                        kl_var.mean()
                    )  # mean of (pi(a|s) / pi_old(a|s) - 1 - log(pi(a|s) / pi_old(a|s)))
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                if args.is_loss_clip:  # clip loss
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:  # KL-penalty loss
                    pg_loss2 = beta * kl_var
                    pg_loss = (pg_loss1 + pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(
                    -1
                )  # value computed by NN with updated parameters
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
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )  # can i implement this myself?
                optimizer.step()

            if args.is_loss_clip:  # if clip loss and approx_kl > target kl, break
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            else:  # if KL-penalty loss, update beta
                beta = (
                    beta / 2
                    if approx_kl < args.target_kl / 1.5
                    else (beta * 2 if approx_kl > args.target_kl * 1.5 else beta)
                )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.wandb_log:
            wandb.log(
                {
                    "charts/global_step": global_step,
                    "charts/episode": episode,
                    "charts/normalized_returns_mean": normalized_returns.mean(),
                    "charts/normalized_lengths_mean": normalized_lengths.mean(),
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/solved": len(success_record["solved"]),
                    "charts/unsolved": len(success_record["unsolved"]),
                    "charts/highest_solved": (
                        max(success_record["solved"])
                        if success_record["solved"]
                        else -1
                    ),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy_loss": entropy_loss.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/explained_variance": explained_var,
                    "losses/clipfrac": np.mean(clipfracs),
                    "debug/advantages_mean": b_advantages.mean(),
                    "debug/advantages_std": b_advantages.std(),
                }
            )

        if update > 0 and update % 100 == 0:  # save a checkpoint every 100 updates
            checkpoint = {
                "critic": agent.critic.state_dict(),
                "actor": agent.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "episode": episode,
                "config": vars(args),
                "mean_return": normalized_returns.mean(),
                "success_record": success_record,
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
                "supermoves": envs.envs[0].supermoves,  # dict of supermoves or None
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, join(out_dir, "ckpt.pt"))

    return
