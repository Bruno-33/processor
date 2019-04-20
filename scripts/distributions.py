import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init, init_normc_, AddBias

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_agents, sigmoid):
        super(DiagGaussian, self).__init__()

        # An instance (i.e., with specific weight and bias initializing function) of the function "init"
        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Number of inputs and outputs for each agent
        self.num_agents = num_agents
        self.atom_num_inputs = num_inputs // num_agents
        self.atom_num_outputs = num_outputs // num_agents
        self.fc_mean = init_(nn.Linear(self.atom_num_inputs, self.atom_num_outputs))
        self.logstd = AddBias(torch.zeros(self.atom_num_outputs))
        self.sigmoid = sigmoid

    def forward(self, x):
        # action_mean:
        #                   |                   |   ...
        # fc_mean(x_{sub0}) | fc_mean(x_{sub1}) |   ...
        #                   |                   |   ...
        action_mean = torch.cat(tuple([self.fc_mean(x[:, i*self.atom_num_inputs:(i+1)*self.atom_num_inputs]) for i in range(self.num_agents)]), dim=1)

        if self.sigmoid:
            action_mean = torch.sigmoid(action_mean)

        #  An ugly hack for my KFAC implementation.
        dummy = action_mean[:, :self.atom_num_outputs]
        zeros = torch.zeros(dummy.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = torch.cat(tuple([self.logstd(zeros) for i in range(self.num_agents)]), dim=1)
        return FixedNormal(action_mean, action_logstd.exp())
