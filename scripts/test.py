import argparse

import numpy as np
import torch
import time

# Arguments
parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--load-dir', required=True,
                    help='directory of models (e.g., PATH_TO_MODEL/model0.pt)')
args = parser.parse_args()

# Model
actor_critic, ob_rms, _ = torch.load(args.load_dir)
actor_critic.cuda()

# Auxiliary variables
state = torch.zeros(1, actor_critic.state_size).cuda()  # Hidden state for RNN
mask = torch.zeros(1, 1).cuda()     # Termination flag

# Observation normalization
num_agents = 1
obs = np.hstack([np.arange(40) for _ in range(num_agents)])
mean = np.hstack([ob_rms.mean for _ in range(num_agents)])
var = np.hstack([ob_rms.var for _ in range(num_agents)])
obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10., 10.)

# Shape adaptation
obs = np.expand_dims(obs, axis=0)
obs = torch.from_numpy(obs).float()     # From CPU to GPU

n = 50
total_time = 0.
for _ in range(n):
    start = time.time() # Compute action
    with torch.no_grad():
        # _, action, _, state = actor_critic.act(torch.Tensor(obs).cuda(), state, mask, deterministic=True)
        action, state = actor_critic.act_exe(torch.Tensor(obs).cuda(), state, mask)
    
    
    # GPU to CPU
    cpu_actions = action.squeeze(0).cpu().numpy()
    if _ != 0:
        total_time += time.time() - start
print(total_time*1000./(n-1))

print(cpu_actions)
