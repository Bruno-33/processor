import argparse
import os
import ctypes
import glob

import numpy as np
import torch
import pandas as pd

import gym
import gym_foa
from gym import wrappers


# Argument stuffs
parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--env-name', required=True,
                    help='environment name')
parser.add_argument('--model-dir', required=True,
                    help='directory of models (e.g., PATH_TO_MODEL/model0.pt)')
parser.add_argument('--num-test', type=int, default=10,
                    help='number of test cases (default: 10)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--indep', action='store_true', default=False,
                    help='switch of independent update (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--scene-dir',
                    help='directory of scenes (e.g., PATH_TO_SCENE)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-render', action='store_true', default=False,
                    help='disables recording')

args = parser.parse_args()
ctypes.cdll.LoadLibrary("libGL.so.1")   # Although ridiculous, only when this function is called before torch.cuda.is_available() can libGL be properly loaded in pyglet
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Path
model_dir, model_name = os.path.dirname(args.model_dir), os.path.basename(args.model_dir)
video_path = os.path.join(model_dir, "video_" + model_name[:-3])   # Suppose model names have suffix .pt
if args.scene_dir:
    data_path = os.path.join(model_dir, "metric_{}.csv".format(model_name[:-3]))   # Suppose model names have suffix .pt

# Environment
env = gym.make(args.env_name)
if not args.no_render:
    env = wrappers.Monitor(env, video_path, lambda episode_id: True, force=True)
env.seed(args.seed)

num_agent = env.unwrapped.hp_uav_n
num_subagents = num_agent if args.indep else 1  # The way you view the robot team (i.e., a virtual structure or many robots)
obs_shape = env.observation_space.shape
atom_obs_shape = (obs_shape[0] // num_subagents * args.num_stack, *obs_shape[1:])

# Agent
actor_critic, ob_rms, _ = \
            torch.load(args.model_dir)

# Auxiliary variables
l_states = [
        torch.zeros(1, actor_critic.state_size)
        for _ in range(num_subagents)
        ]
masks = torch.zeros(1, 1)
current_obs = [
        torch.zeros(1, *atom_obs_shape)
        for _ in range(num_subagents)
        ]

if args.cuda:
    actor_critic.cuda()
    masks = masks.cuda()
    for i in range(num_subagents):
        l_states[i] = l_states[i].cuda()
        current_obs[i] = current_obs[i].cuda()


# Stack sequent observations to get current_obs, using the trick of reshaping.
#
# current_obs
# Index         |1           |2           |3
# Observation   |a1 a2 a3    |b1 b2 b3    |c1 c2 c3
def update_current_obs(obs, idx):
    shape_dim0 = atom_obs_shape[0] // args.num_stack
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[idx][:, :-shape_dim0] = current_obs[idx][:, shape_dim0:]
    current_obs[idx][:, -shape_dim0:] = obs


# Quantitative metric
title = {'success_flag': [], 'efficiency': [], 'connectivity': [], 'smoothness': [],}
df = pd.DataFrame(title)

# Unify loading and randomly generating scenes with iterator
if args.scene_dir:
    iterator = glob.glob(os.path.join(args.scene_dir, 'scene*.pkl'))   # A list of scenes
else:
    iterator = range(args.num_test)

fail_scenes = []
for it in iterator:
    if args.scene_dir:  # Load scenes when args.scene_dir is provided
        env.unwrapped.load_scene(it)
    obs = env.reset()

    while True:
        # Observation normalization
        atom_policy_obs_shape = (obs_shape[0] // num_agent * args.num_stack, *obs_shape[1:])    # Different from atom_obs_shape, use num_agent rather than num_subagents
        if len(ob_rms.mean) == atom_policy_obs_shape[0]:  # Same mean & var for different agents
            mean = np.zeros(len(ob_rms.mean) * num_agent)
            var = np.zeros(len(ob_rms.var) * num_agent)
            for i in range(num_agent):
                mean[i*atom_policy_obs_shape[0]:(i+1)*atom_policy_obs_shape[0]] = ob_rms.mean
                var[i*atom_policy_obs_shape[0]:(i+1)*atom_policy_obs_shape[0]] = ob_rms.var
        elif len(ob_rms.mean) == atom_policy_obs_shape[0] * num_agent:  # Different mean & var for different uavs
            mean = ob_rms.mean
            var = ob_rms.var
        else:
            raise RuntimeError("Mismatched num_stack")

        obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10., 10.)
        obs = np.expand_dims(obs, axis=0)

        for i in range(num_subagents):
            current_obs[i] *= masks     # Useful when args.num_stack > 1
            update_current_obs(obs[:, i*atom_obs_shape[0]:(i+1)*atom_obs_shape[0]], i)

        # Interaction with the environment
        with torch.no_grad():
            l_action = []
            for i in range(num_subagents):
                _, action, _, l_states[i] = actor_critic.act(
                        current_obs[i],
                        l_states[i],
                        masks,
                        deterministic=True)
                l_action.append(action)
            action = torch.cat(l_action, dim=1)

        cpu_actions = action.squeeze(0).cpu().numpy()

        obs, _, done, info = env.step(cpu_actions)

        masks.fill_(0.0 if done else 1.0)

        if done:
            if not args.no_render:
                env.video_recorder.capture_frame()  # Save the last frame
            break

    metric = pd.Series([info['success'], info['efficiency'], info['connectivity'], info['smoothness']],
                     index=['success_flag', 'efficiency', 'connectivity', 'smoothness',])
    df = df.append(metric, ignore_index=True)
    if not info['success']:
        fail_scenes.append(it)

print(df)
if args.scene_dir:
    df.to_csv(data_path)
    print("Fail in the following scenes:")
    print("\n".join(fail_scenes))
