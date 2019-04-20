import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_
import copy


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(
            self, obs_shape, action_shape, sensor_type, atom_num_inputs_o_env, dim, num_agents,
            unordered, independent, sigmoid, share, no_rnn
            ):
        super(Policy, self).__init__()

        # Base network
        assert len(obs_shape) == 1, "We only handle flattened input."
        self.base = RNNBase(obs_shape[0], sensor_type, atom_num_inputs_o_env, dim, num_agents, unordered, independent, share, no_rnn)

        # Actor's final layer
        num_outputs = action_shape[0]
        if independent:
            self.dist = DiagGaussian(self.base.output_size, num_outputs, 1, sigmoid)
        else:
            self.dist = DiagGaussian(self.base.output_size, num_outputs, num_agents, sigmoid)

        self.state_size = self.base.state_size
        self.sigmoid = sigmoid

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

    # CAUTION: Remember to update the self.base.forward_exe when self.base.forward is updated
    def act_exe(self, inputs, states, masks):   # For decentralized execution (block the critic, only run one channel of the actor)
        self.base.num_subnets = 1
        self.dist.num_agents = 1
        actor_features, states = self.base.forward_exe(inputs, states, masks)
        dist = self.dist(actor_features)
        action = dist.mode()

        return action, states


class RNNBase(nn.Module):
    # Parameters:
    #   share: Share observation feature extraction modules between actor and critic
    #   no_rnn: Delete GRU
    def __init__(self, num_inputs, sensor_type, atom_num_inputs_o_env, dim, num_agents, unordered, independent, share, no_rnn):
        super(RNNBase, self).__init__()

        # Handling arguments
        self.sensor_type = sensor_type
        self.atom_num_inputs_o_env = atom_num_inputs_o_env
        self.dim = dim
        self.num_agents = num_agents
        self.unordered = unordered
        self.num_subnets = 1 if independent else self.num_agents     # Number of subnetworks
        self.atom_num_inputs = num_inputs // self.num_subnets
        self.no_rnn = no_rnn

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Network architecture parameters
        cnn1_channel, cnn1_kernel, cnn1_stride = 32, 5, 2
        cnn2_channel, cnn2_kernel, cnn2_stride = 32, 3, 2
        cnn_out = (self.atom_num_inputs_o_env - cnn1_kernel) // cnn1_stride + 1
        cnn_out = (cnn_out - cnn2_kernel) // cnn2_stride + 1
        cnn_out *= cnn2_channel     # Size of flattened CNN layer output
        fc1 = 256
        fc2 = 128   # The last layer of feature extraction module
        fc3 = 128   # The last layer before GRU

        # Feature extraction module

        # Additional module to achieve unordered o_partner
        if self.unordered:
            fc_o_partner = 64   # unordered o_partner enabling layer
            self.critic_o_partner = nn.Sequential(
                init_(nn.Linear(self.dim, fc_o_partner)),
                nn.ReLU()
                )
            self.actor_o_partner = self.critic_o_partner if share else copy.deepcopy(self.critic_o_partner)

        if self.sensor_type == "lidar":
            # Critic
            self.critic_cnn = nn.Sequential(
                init_(nn.Conv1d(1, cnn1_channel, cnn1_kernel, stride=cnn1_stride)),
                nn.ReLU(),
                init_(nn.Conv1d(cnn1_channel, cnn2_channel, cnn2_kernel, stride=cnn2_stride)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(cnn_out, fc1)),
                nn.ReLU()
                )
            # Input size of the next fc layer
            if self.unordered:
                fc_in = (self.atom_num_inputs
                        - self.atom_num_inputs_o_env + fc1  # o_env
                        - self.dim*(self.num_agents-1) + fc_o_partner)   # o_partner
            else:
                fc_in = (self.atom_num_inputs
                        - self.atom_num_inputs_o_env + fc1)
            self.critic_prep = nn.Sequential(
                init_(nn.Linear(fc_in, fc2)),
                nn.ReLU()
                )

            # Actor
            self.actor_cnn = self.critic_cnn if share else copy.deepcopy(self.critic_cnn)
            self.actor_prep = self.critic_prep if share else copy.deepcopy(self.critic_prep)
        elif self.sensor_type == "pos":
            # Critic
            # Input size of the next fc layer
            if self.unordered:
                fc_in = (self.atom_num_inputs
                        - self.dim*(self.num_agents-1) + fc_o_partner)   # o_partner
            else:
                fc_in = self.atom_num_inputs
            self.critic_prep = nn.Sequential(
                init_(nn.Linear(fc_in, fc2)),
                nn.ReLU()
                )

            # Actor
            self.actor_prep = self.critic_prep if share else copy.deepcopy(self.critic_prep)
        else:
            raise NotImplementedError

        # Subsequent modules
        # Critic
        self.critic_rnn_prep = nn.Sequential(
            init_(nn.Linear(fc2, fc3)),
            nn.ReLU()
            )
        # Actor
        self.actor_rnn_prep = copy.deepcopy(self.critic_rnn_prep)

        self.gru = "hack"   # The original code decides the type of data_generator with hasattr(self, 'gru')

        if not self.no_rnn:
            # Critic
            self.critic_rnn = nn.GRUCell(fc3, fc3)
            nn.init.orthogonal_(self.critic_rnn.weight_ih.data)
            nn.init.orthogonal_(self.critic_rnn.weight_hh.data)
            self.critic_rnn.bias_ih.data.fill_(0)
            self.critic_rnn.bias_hh.data.fill_(0)

            # Actor
            self.actor_rnn = copy.deepcopy(self.critic_rnn)
        else:   # Compensating layers to have a similar network structure
            self.critic_proxy_rnn = init_(nn.Linear(fc3, fc3))
            self.actor_proxy_rnn = copy.deepcopy(self.critic_proxy_rnn)

        self.critic_linear = nn.Sequential(
            init_(nn.Linear(fc3, fc3)),
            init_(nn.Linear(fc3, 1))
            )

        self.train()

    @property
    def state_size(self):
        if not self.no_rnn:
            # [h_{critic}, h_{agent1}, h_{agent2}, ...]
            return self.critic_rnn.hidden_size + self.actor_rnn.hidden_size*self.num_subnets
        else:
            return 1

    @property
    def output_size(self):
        if not self.no_rnn:
            return self.actor_rnn.hidden_size*self.num_subnets
        else:
            return self.actor_proxy_rnn.out_features*self.num_subnets

    def forward(self, inputs, states, masks):
        # Feature extraction module
        for i in range(self.num_subnets):
            # Bounders of different observations
            o_env_l, o_env_r = i*self.atom_num_inputs, i*self.atom_num_inputs + self.atom_num_inputs_o_env
            o_partner_l, o_partner_r = o_env_r, o_env_r + self.dim*(self.num_agents-1)

            # Observations of each agent
            o_env_i = inputs[:, o_env_l:o_env_r]
            o_partner_i = inputs[:, o_partner_l:o_partner_r]
            # o_partner_i = o_partner_i.squeeze(1)
            o_else_i = inputs[:, o_partner_r:(i+1)*self.atom_num_inputs]
            # o_else_i = o_else_i.squeeze(1)

            # Process o_env_i
            if self.sensor_type == "lidar":
                o_env_i = o_env_i.unsqueeze(1)
                # Critic
                o_env_c_i = self.critic_cnn(o_env_i)
                # Actor
                o_env_a_i = self.actor_cnn(o_env_i)
            elif self.sensor_type == "pos":
                o_env_c_i = o_env_i
                o_env_a_i = o_env_i
            else:
                raise NotImplementedError

            # Process o_partner_i
            if self.unordered:
                # sum(fc(p0), fc(p1), ...)
                for j in range(self.num_agents-1):
                    p_j = o_partner_i[:, j*self.dim:(j+1)*self.dim]    # The j-th partner
                    # Critic
                    p_c_j = self.critic_o_partner(p_j)
                    o_partner_c_i = p_c_j if j==0 else o_partner_c_i + p_c_j
                    # Actor
                    p_a_j = self.actor_o_partner(p_j)
                    o_partner_a_i = p_a_j if j==0 else o_partner_a_i + p_a_j
            else:
                o_partner_c_i = o_partner_i
                o_partner_a_i = o_partner_i

            # Final
            x_c_i = torch.cat((o_env_c_i, o_partner_c_i, o_else_i), dim=1)
            x_a_i = torch.cat((o_env_a_i, o_partner_a_i, o_else_i), dim=1)

            # Critic
            x_c_i = self.critic_prep(x_c_i)  # Output of feature extraction module
            x_c = x_c_i if i==0 else x_c + x_c_i
            # Actor
            x_a_i = self.actor_prep(x_a_i)
            x_a_i = self.actor_rnn_prep(x_a_i)
            x_a = x_a_i if i==0 else torch.cat((x_a, x_a_i), dim=1) # In actor, each subnetwork never interleaves

        x_c = self.critic_rnn_prep(x_c) # In critic, features are added up before further process

        if not self.no_rnn:
            # RNN stuffs

            rnn_size_in_a = self.actor_rnn.input_size
            rnn_size_out_a = self.actor_rnn.hidden_size
            s_c = states[:, :-rnn_size_out_a*self.num_subnets]
            s_a = states[:, -rnn_size_out_a*self.num_subnets:]

            # During execution, size of inputs matchs size of states
            if inputs.size(0) == states.size(0):
                x_c = s_c = self.critic_rnn(x_c, s_c * masks)
                x_a = s_a = torch.cat(
                        tuple([self.actor_rnn(
                            x_a[:, i*rnn_size_in_a:(i+1)*rnn_size_in_a],
                            s_a[:, i*rnn_size_out_a:(i+1)*rnn_size_out_a] * masks)
                            for i in range(self.num_subnets)]),
                        dim=1)
            # During parameter update procedure, actions are computed with new parameters.
            # Hence, only states at the first time step are required.  Actually, I think it
            # should always be in the initial states at the first time step.  But in this
            # implementation, this is not guaranteed.  As a result, sizes of states and
            # inputs are different.
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x_c.size(0) / N)

                # unflatten
                x_c = x_c.view(T, N, x_c.size(1))
                x_a = x_a.view(T, N, x_a.size(1))

                # Same deal with masks
                masks = masks.view(T, N, 1)

                o_c = []
                o_a = []
                for t in range(T):
                    hx_c = s_c = self.critic_rnn(x_c[t], s_c * masks[t])
                    hx_a = s_a = torch.cat(
                            tuple([self.actor_rnn(
                                x_a[t, :, i*rnn_size_in_a:(i+1)*rnn_size_in_a],
                                s_a[:, i*rnn_size_out_a:(i+1)*rnn_size_out_a] * masks[t])
                                for i in range(self.num_subnets)]),
                            dim=1)
                    o_c.append(hx_c)    # Why bother creating a new variable hx_c???
                    o_a.append(hx_a)

                # x is a (T, N, -1) tensor
                x_c = torch.stack(o_c, dim=0)
                x_a = torch.stack(o_a, dim=0)

                # flatten
                x_c = x_c.view(T * N, -1)
                x_a = x_a.view(T * N, -1)

            return self.critic_linear(x_c), x_a, torch.cat([s_c, s_a], dim=1)
        else:
            x_c = self.critic_proxy_rnn(x_c)
            proxy_rnn_size_in_a = self.actor_proxy_rnn.in_features
            x_a = torch.cat(
                    tuple([self.actor_proxy_rnn(
                        x_a[:, i*proxy_rnn_size_in_a:(i+1)*proxy_rnn_size_in_a])
                        for i in range(self.num_subnets)]),
                    dim=1)
            return self.critic_linear(x_c), x_a, states

    def forward_exe(self, inputs, states, masks):
        # Feature extraction module
        for i in range(self.num_subnets):
            # Bounders of different observations
            o_env_l, o_env_r = i*self.atom_num_inputs, i*self.atom_num_inputs + self.atom_num_inputs_o_env
            o_partner_l, o_partner_r = o_env_r, o_env_r + self.dim*(self.num_agents-1)

            # Observations of each agent
            o_env_i = inputs[:, o_env_l:o_env_r]
            o_partner_i = inputs[:, o_partner_l:o_partner_r]
            # o_partner_i = o_partner_i.squeeze(1)
            o_else_i = inputs[:, o_partner_r:(i+1)*self.atom_num_inputs]
            # o_else_i = o_else_i.squeeze(1)

            # Process o_env_i
            if self.sensor_type == "lidar":
                o_env_i = o_env_i.unsqueeze(1)
                # Actor
                o_env_a_i = self.actor_cnn(o_env_i)
            elif self.sensor_type == "pos":
                o_env_a_i = o_env_i
            else:
                raise NotImplementedError

            # Process o_partner_i
            if self.unordered:
                # sum(fc(p0), fc(p1), ...)
                for j in range(self.num_agents-1):
                    p_j = o_partner_i[:, j*self.dim:(j+1)*self.dim]    # The j-th partner
                    # Actor
                    p_a_j = self.actor_o_partner(p_j)
                    o_partner_a_i = p_a_j if j==0 else o_partner_a_i + p_a_j
            else:
                o_partner_a_i = o_partner_i

            # Final
            x_a_i = torch.cat((o_env_a_i, o_partner_a_i, o_else_i), dim=1)

            # Actor
            x_a_i = self.actor_prep(x_a_i)
            x_a_i = self.actor_rnn_prep(x_a_i)
            x_a = x_a_i if i==0 else torch.cat((x_a, x_a_i), dim=1) # In actor, each subnetwork never interleaves

        if not self.no_rnn:
            # RNN stuffs

            rnn_size_in_a = self.actor_rnn.input_size
            rnn_size_out_a = self.actor_rnn.hidden_size
            s_a = states[:, -rnn_size_out_a*self.num_subnets:]

            # During execution, size of inputs matchs size of states
            if inputs.size(0) == states.size(0):
                x_a = s_a = torch.cat(
                        tuple([self.actor_rnn(
                            x_a[:, i*rnn_size_in_a:(i+1)*rnn_size_in_a],
                            s_a[:, i*rnn_size_out_a:(i+1)*rnn_size_out_a] * masks)
                            for i in range(self.num_subnets)]),
                        dim=1)
            # During parameter update procedure, actions are computed with new parameters.
            # Hence, only states at the first time step are required.  Actually, I think it
            # should always be in the initial states at the first time step.  But in this
            # implementation, this is not guaranteed.  As a result, sizes of states and
            # inputs are different.
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x_c.size(0) / N)

                # unflatten
                x_a = x_a.view(T, N, x_a.size(1))

                # Same deal with masks
                masks = masks.view(T, N, 1)

                o_a = []
                for t in range(T):
                    hx_a = s_a = torch.cat(
                            tuple([self.actor_rnn(
                                x_a[t, :, i*rnn_size_in_a:(i+1)*rnn_size_in_a],
                                s_a[:, i*rnn_size_out_a:(i+1)*rnn_size_out_a] * masks[t])
                                for i in range(self.num_subnets)]),
                            dim=1)
                    o_a.append(hx_a)

                # x is a (T, N, -1) tensor
                x_a = torch.stack(o_a, dim=0)

                # flatten
                x_a = x_a.view(T * N, -1)

            return x_a, torch.cat([s_a], dim=1)
        else:
            proxy_rnn_size_in_a = self.actor_proxy_rnn.in_features
            x_a = torch.cat(
                    tuple([self.actor_proxy_rnn(
                        x_a[:, i*proxy_rnn_size_in_a:(i+1)*proxy_rnn_size_in_a])
                        for i in range(self.num_subnets)]),
                    dim=1)
            return x_a, states
