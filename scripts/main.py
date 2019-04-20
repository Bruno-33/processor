import ctypes
import copy
import os
import time
import datetime

import gym
import gym_foa
import numpy as np
import torch

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import Policy
from storage import RolloutStorage
from visualize import visdom_plot
import algo
from logger import Logger


ctypes.cdll.LoadLibrary("libGL.so.1")   # Although ridiculous, only when this function is called before torch.cuda.is_available() can libGL be properly loaded in pyglet
args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Path stuffs
setting = []
update_rule = "indep" if args.indep else "clde"
order = "unordered" if args.unordered else "ordered"
sigmoid = "sigmoid" if args.sigmoid else "fc"
share = "share" if args.share else "seperate"
no_rnn = "no-rnn" if args.no_rnn else "rnn"
setting.append(update_rule)
setting.append(order)
setting.append(sigmoid)
setting.append(share)
setting.append(no_rnn)
setting = "_".join(setting)
title = "\n".join([args.env_name[:50], setting, "seed" + str(args.seed)])

path_suffix = os.path.join(args.env_name, setting, "seed" + str(args.seed))
root_path = os.path.join(args.log_dir, path_suffix)
log_path = os.path.join(root_path, "log")   # For Visdom and Tensorboard
model_path = os.path.join(root_path, "model")   # For models
# For safety, let it crush if paths exist
os.makedirs(log_path)
os.makedirs(model_path)

# logger = Logger(log_path)  # For Tensorboard

def main():
    torch.set_num_threads(1)

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    # Environment stuffs

    envs = []
    for i in range(args.num_processes):
        if args.scene_dir:
            scene_dir = os.path.join(args.scene_dir, "seed{}".format(args.seed+i))
            assert os.path.exists(scene_dir)
        else:
            scene_dir = None
        envs.append(make_env(args.env_name, args.seed, i, log_path, args.add_timestep, scene_dir))

    # Hack infomation of gym environment
    tmp_env = envs[0]()
    sensor_type = tmp_env.unwrapped.hp_sensing_mode
    num_agent = tmp_env.unwrapped.hp_uav_n
    dim = tmp_env.unwrapped.hp_dim
    # Shape of o_env for each agent, required by the observation feature extraction module of the model
    if sensor_type == "lidar":
        atom_o_env_shape = tmp_env.unwrapped.hp_lidar_n + dim
    elif sensor_type == "pos":
        atom_o_env_shape = (dim+1) * tmp_env.unwrapped.hp_n_nearest_obs
    else:
        raise Exception("No implementation for sensing mode {}".format(sensor_type))

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if not args.unordered:
            envs = VecNormalize(envs, gamma=args.gamma) # Different observation normalization factors for different agents
        else:
            envs = VecNormalize(envs, gamma=args.gamma, num_agent=num_agent)

    num_subagents = num_agent if args.indep else 1  # The way you view the robot team (i.e., a virtual structure or many robots)
    obs_shape = envs.observation_space.shape
    atom_obs_shape = (obs_shape[0] // num_subagents * args.num_stack, *obs_shape[1:])   # Shape for each logical agent

    action_shape = envs.action_space.shape
    atom_action_shape = (action_shape[0] // num_subagents, *action_shape[1:])

    # Agent stuffs (core elements of PPO)

    if args.load_dir:   # Resume from breakpoint
        print("Loading model parameters from: " + args.load_dir)
        actor_critic, ob_rms, ret_rms = torch.load(args.load_dir)
        assert envs.ob_rms.mean.shape == ob_rms.mean.shape, "Mismatched observation shape, which may be induced by wrong flags (e.g., --unordered / --num_stack)"
        envs.ob_rms = ob_rms
        envs.ret_rms = ret_rms
    else:
        actor_critic = Policy(atom_obs_shape, atom_action_shape, sensor_type, atom_o_env_shape, dim, num_agent, args.unordered, args.indep, args.sigmoid, args.share, args.no_rnn)

    if args.cuda:
        actor_critic.cuda()

    agent = algo.PPO(
            actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
            args.value_loss_coef, args.entropy_coef, lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm
            )

    rollouts = [
            RolloutStorage(args.num_steps, args.num_processes, atom_obs_shape, atom_action_shape, actor_critic.state_size)
            for _ in range(num_subagents)
            ]

    # Auxiliary stuffs

    current_obs = [
            torch.zeros(args.num_processes, *atom_obs_shape)
            for _ in range(num_subagents)
            ]

    # Stack sequent observations to get current_obs, using the trick of reshaping.
    #
    # current_obs
    # Index         |1           |2           |3
    # Observation   |a1 a2 a3    |b1 b2 b3    |c1 c2 c3
    def update_current_obs(obs, idx):
        nonlocal current_obs
        shape_dim0 = atom_obs_shape[0] // args.num_stack
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[idx][:, :-shape_dim0] = current_obs[idx][:, shape_dim0:]
        current_obs[idx][:, -shape_dim0:] = obs

    obs = envs.reset()
    for i in range(num_subagents):
        update_current_obs(obs[:, i*atom_obs_shape[0]:(i+1)*atom_obs_shape[0]], i)
        rollouts[i].observations[0].copy_(current_obs[i])

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        for i in range(num_subagents):
            current_obs[i] = current_obs[i].cuda()
            rollouts[i].cuda()

    # Main loop

    train_start = datetime.datetime.now()
    print("Training starts at: {}".format(train_start))
    env_time = 0.   # time cost of interaction with environment
    env_compute_time = 0.
    env_step_time = 0.
    env_rollout_time = 0.
    update_time = 0.    # time cost of updating parameters
    log_time = 0.   # time cost of logging

    for j in range(num_updates):
        # Interact with the environment

        start_env_time = time.time()    # Timer

        for step in range(args.num_steps):
            start_env_compute_time = time.time()

            # Sample actions
            with torch.no_grad():
                l_value, l_action, l_action_log_prob, l_states = [], [], [], []
                for i in range(num_subagents):
                    value, action, action_log_prob, states = actor_critic.act(
                            rollouts[i].observations[step],
                            rollouts[i].states[step],
                            rollouts[i].masks[step])
                    l_value.append(value)
                    l_action.append(action)
                    l_action_log_prob.append(action_log_prob)
                    l_states.append(states)
                action = torch.cat(l_action, dim=1)

            cpu_actions = action.squeeze(1).cpu().numpy()

            env_compute_time += time.time() - start_env_compute_time

            start_env_step_time = time.time()

            obs, reward, done, info = envs.step(cpu_actions)

            env_step_time += time.time() - start_env_step_time

            start_env_rollout_time = time.time()

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            # final_rewards is the accumulated reward of the last trajectory, episode_rewards is an auxuliary variable.
            # The motivation is to enable logging in arbitrary time step.
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards  # If not done, mask=1, final_rewards doesn't change
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            for i in range(num_subagents):
                current_obs[i] *= masks     # Useful when args.num_stack > 1
                update_current_obs(obs[:, i*atom_obs_shape[0]:(i+1)*atom_obs_shape[0]], i)
                rollouts[i].insert(
                        current_obs[i], l_states[i], l_action[i],
                        l_action_log_prob[i], l_value[i], reward, masks
                        )

            env_rollout_time += time.time() - start_env_rollout_time

        env_time += time.time() - start_env_time

        # Update parameters

        start_update_time = time.time()     # Timer

        for i in range(num_subagents):
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts[i].observations[-1],
                                                    rollouts[i].states[-1],
                                                    rollouts[i].masks[-1]).detach()

            rollouts[i].compute_returns(next_value, args.use_gae, args.gamma, args.tau)

            value_loss, action_loss, dist_entropy = agent.update(rollouts[i])

            rollouts[i].after_update()

        update_time += time.time() - start_update_time

        # Logging

        start_log_time = time.time()    # Timer

        # Save models
        if j % args.save_interval == 0 or j == num_updates-1:
            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs, 'ob_rms') and envs.ob_rms or None,
                          hasattr(envs, 'ret_rms') and envs.ret_rms or None]

            torch.save(save_model, os.path.join(model_path, "model" + str(j) + ".pt"))

        # For logging training information
        if j % args.log_interval == 0 or j == num_updates-1:
            log_env_time = []
            for i, info_i in enumerate(info):
                log_reset_i = "            Average reset time for env{}: {:.1f}ms = {:.1f}h / {}".format(
                        i, info_i['reset_time'] * 1000 / info_i['reset_num'], info_i['reset_time'] / 3600, info_i['reset_num']
                        )
                log_step_i = "            Average step time for env{}: {:.1f}ms = {:.1f}h / {}".format(
                        i, info_i['step_time'] * 1000 / info_i['step_num'], info_i['step_time'] / 3600, info_i['step_num']
                        )
                log_env_time.append(log_reset_i)
                log_env_time.append(log_step_i)
            log_env_time = '\n'.join(log_env_time)

            current_time = datetime.datetime.now()

            summary = '\n'.join([
                    "Training starts at: {}".format(train_start),
                    "Current time: {}".format(current_time),
                    "Elapsed time: {}".format(current_time-train_start),
                    "    Environment interaction: {:.1f}h".format(env_time/3600),
                    "        Compute action: {:.1f}h".format(env_compute_time/3600),
                    "        Rollout: {:.1f}h".format(env_rollout_time/3600),
                    "        Interaction with gym: {:.1f}h".format(env_step_time/3600),
                    log_env_time,
                    "    Parameters update: {:.1f}h".format(update_time/3600),
                    "    logging: {:.1f}h".format(log_time/3600)
                    ]) + '\n'

            # Write down summary of the training
            with open(os.path.join(root_path, "summary.txt"), 'w') as f:
                f.write(summary)

        # For Visdom visualization
        if args.vis and (j % args.vis_interval == 0 or j == num_updates-1):
            # Sometimes monitor doesn't properly flush the outputs
            win = visdom_plot(viz, win, args.vis_env, log_path, title,
                              args.algo, args.num_frames, save_dir=root_path)
            viz.save([args.vis_env])

        log_time += time.time() - start_log_time

    print(summary)


if __name__ == "__main__":
    main()
