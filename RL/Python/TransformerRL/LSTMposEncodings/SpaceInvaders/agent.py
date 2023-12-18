import os
import torch as T
import torch.nn.functional as F
import numpy as np
from transformer_network_posLSTM_SpreadAttention import *
from buffer import *


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.999, update_actor_interval=2, warmup=1000,
                 n_actions=4, max_size=1000000, layer1_size=64,
                 layer2_size=32, batch_size=100, noise=0.01):
        self.gamma = gamma
        self.tau = tau
        self.max_action = 10
        self.min_action = -10
        # Replay buffer now uses the updated ReplayBuffer class for PER
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')



        self.actor = ActorNetwork(alpha, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=n_actions, name='actor_LSTM', chkpt_dir='td3_MAT').to(self.device)
        self.critic_1 = CriticNetwork(beta, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=n_actions,
            name='critic_1_LSTM', chkpt_dir='td3_MAT').to(self.device)
        self.critic_2 = CriticNetwork(beta, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=n_actions,
            name='critic_2_LSTM', chkpt_dir='td3_MAT').to(self.device)

        self.target_actor = ActorNetwork(alpha, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=n_actions, name='target_actor_LSTM', chkpt_dir='td3_MAT').to(self.device)
        self.target_critic_1 = CriticNetwork(beta, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=n_actions,
            name='target_critic_1_LSTM', chkpt_dir='td3_MAT').to(self.device)
        self.target_critic_2 = CriticNetwork(beta, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=n_actions,
            name='target_critic_2_LSTM', chkpt_dir='td3_MAT').to(self.device)

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, mask):
        if self.time_step < self.warmup*0:
            print(self.time_step,self.warmup*1)
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mask = T.tensor(mask, dtype=T.float).to(self.actor.device)

            mu = self.actor.forward(state,mask).to(self.actor.device)[0]
            # print('mu', mu)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=mu.shape), dtype=T.float).to(self.device)
        mu_prime[3:] = mu[3:] + T.tensor(np.random.normal(scale=self.noise*10, size=16), dtype=T.float).to(self.device)

        # print('mu prime',mu_prime)
        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, mask, action, reward, new_state, next_mask, done):
        self.memory.store_transition(state, mask, action, reward, new_state, next_mask, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size*1:
            return

        state, mask, action, reward, new_state, next_mask, done, indices, weights = \
            self.memory.sample_buffer(self.batch_size)

        # state, action, reward, new_state, done = \
        #     self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        mask_ = T.tensor(next_mask, dtype=T.float).to(self.critic_1.device)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        mask = T.tensor(mask, dtype=T.float).to(self.critic_1.device)

        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_, mask_)
        # target_actions = target_actions + \
        #                  T.clamp(T.tensor(np.random.normal(scale=0.1)), -0.5, 0.5)
        noise = T.clamp(T.tensor(np.random.normal(scale=0.001, size=target_actions.shape), dtype=T.float), -0.5, 0.5).to(self.device)
        target_actions = target_actions + noise

        # might break if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action, self.max_action)

        q1_ = self.target_critic_1.forward(state_, target_actions, mask_)
        q2_ = self.target_critic_2.forward(state_, target_actions, mask_)

        q1 = self.critic_1.forward(state, action, mask)
        q2 = self.critic_2.forward(state, action, mask)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)


        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()


        # Convert logits to probabilities
        logits = self.actor.forward(state, mask)  # assuming this outputs logits
        # probabilities = F.softmax(logits[:,0:4], dim=-1)

        # Calculate entropy
        # entropy = -(probabilities * torch.log(probabilities + 1e-6)).sum(-1).mean()
        # entropy_coefficient = 0.0001  # This is a hyperparameter you can tune

        # Calculate the L2 norm of the logits (regularization term)
        logits_l2_norm = T.mean(logits[:,:] ** 2)
        regularization_coefficient = 0.001  # This is a hyperparameter you can tune

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, logits, mask)
        actor_loss = -T.mean(actor_q1_loss) + regularization_coefficient * logits_l2_norm
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_network_parameters()

        with T.no_grad():
            # Compute TD errors for updating priorities
            # TD error is the absolute difference between target Q values and current Q values
            td_error = T.abs(target - T.min(q1, q2)).detach().cpu().numpy()
            # Update priorities in the replay buffer
            # print('ayo',indices.shape)
            # print(td_error.shape)
            self.memory.update_priorities(indices, td_error)


    # def update_priorities(self, indices, errors):
    #     # Flatten or squeeze the errors array to make it one-dimensional
    #     errors = np.squeeze(errors)
    #     # Update the priorities for the sampled experiences in the buffer
    #     for idx, error in zip(indices, errors):
    #         self.memory.update_priorities(idx, error)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)



        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                                        (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

