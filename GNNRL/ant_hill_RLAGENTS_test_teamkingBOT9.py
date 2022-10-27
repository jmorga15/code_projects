# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:49:38 2022

@author: jomor
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:23:40 2022

@author: jomor
"""
import numpy as np
from matplotlib import pyplot as plt

class ant_hill():
    def init(self, env_size = 16):


        ### initialize environment settings ###
        self.boxL = 16
        self.boxH = 16
        self.target_location = np.array([np.random.randint(self.boxL, size=1),np.random.randint(self.boxH , size=1)])

        print('Target Location', (self.target_location[0],self.target_location[1]))
        ### actor start (bottom left corner) ###
        ### 4 ants (x,y, hill_king) ###
        ### hill king denotes whether ant is in the hill -> 0,1
        self.ant_locations = np.zeros((8,3))

        for i in range(len(self.ant_locations)):
            self.ant_locations[i,0] = np.random.randint(self.boxL, size=1)[0]
            self.ant_locations[i,1] = np.random.randint(self.boxH, size=1)[0]

        ### the adjacency graph per iteration ###
        self.adjacency = np.zeros((8,8))

        ### compute distance ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]
            for j in range(len(self.ant_locations)):
                xj = self.ant_locations[j,0]
                yj = self.ant_locations[j,1]

                radial_distance = np.sqrt( (xi - xj)**2 + (yi - yj)**2 )

                if radial_distance < 6:
                    self.adjacency[i,j] = 1
                    self.adjacency[j,i] = 1
                else:
                    self.adjacency[i,j] = 0
                    self.adjacency[j,i] = 0

        print("Adjacency Matrix", self.adjacency)

        self.t = 0

        self.done=False


        return self.ant_locations, self.adjacency

    def reset(self, d_message = 64):

        ### initialize environment settings ###
        self.boxL = 16
        self.boxH = 16
        self.target_location = np.array([np.random.randint(self.boxL, size=1),np.random.randint(self.boxH , size=1)])

        print('Target Location', (self.target_location[0],self.target_location[1]))
        ### actor start (bottom left corner) ###
        ### 4 ants (x,y, hill_king) ###
        ### hill king denotes whether ant is in the hill -> 0,1
        self.ant_locations = np.zeros((8,3))

        for i in range(len(self.ant_locations)):
            self.ant_locations[i,0] = np.random.randint(self.boxL, size=1)[0]
            self.ant_locations[i,1] = np.random.randint(self.boxH, size=1)[0]

        ### the adjacency graph per iteration ###
        self.adjacency = np.zeros((8,8))

        ### compute distance ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]
            for j in range(len(self.ant_locations)):
                xj = self.ant_locations[j,0]
                yj = self.ant_locations[j,1]

                radial_distance = np.sqrt( (xi - xj)**2 + (yi - yj)**2 )

                if radial_distance < 6:
                    self.adjacency[i,j] = 1
                    self.adjacency[j,i] = 1
                else:
                    self.adjacency[i,j] = 0
                    self.adjacency[j,i] = 0

        print("Adjacency Matrix", self.adjacency)

        self.t = 0

        self.done=False




        return self.ant_locations, self.adjacency


    def step(self, ant_actions):

        ### ant actions is 4x5 matrix
        ### 4 ants by 5 possible actions
        ### actions (right,left,up,down)
        self.reward = 0
        ### update ant_locations ###
        for i in range(len(ant_actions)):
            if ant_actions[i,0] == 1:
                ant_loc_update = [1,0]
            elif ant_actions[i,1] == 1:
                ant_loc_update = [-1,0]
            elif ant_actions[i,2] == 1:
                ant_loc_update = [0,1]
            elif ant_actions[i,3] == 1:
                ant_loc_update = [0,-1]
            elif ant_actions[i,4] == 1:
                ant_loc_update = [0,0]

            if self.ant_locations[i,0]  + ant_loc_update[0] >= 0 and  self.ant_locations[i,0] + ant_loc_update[0] <= self.boxL:
                self.ant_locations[i,0] = self.ant_locations[i,0] + ant_loc_update[0]
            else:
                self.reward += -0.0

            if self.ant_locations[i,1] + ant_loc_update[1] >= 0 and  self.ant_locations[i,1] + ant_loc_update[1] <= self.boxH:
                self.ant_locations[i,1] = self.ant_locations[i,1] + ant_loc_update[1]

            else:
                self.reward += -0.0


        ### update ant_rewards ###
        ### each turn, each ant receives a reward which is the sum of all ants in the hill ###

        #self.reward = 0

        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]

            targx = self.target_location[0]
            targy = self.target_location[1]

            radial_distance = np.sqrt( (xi - targx)**2 + (yi - targy)**2 )

            if radial_distance < 2:
                self.ant_locations[i,2]=1
                self.reward += 1
            else:
                self.ant_locations[i,2]=0

        ### compute new adjacency matrix ###
        ### compute distance ###
        for i in range(len(self.ant_locations)):
            xi = self.ant_locations[i,0]
            yi = self.ant_locations[i,1]
            for j in range(len(self.ant_locations)):
                xj = self.ant_locations[j,0]
                yj = self.ant_locations[j,1]

                radial_distance = np.sqrt( (xi - xj)**2 + (yi - yj)**2 )

                if radial_distance < 8:
                    self.adjacency[i,j] = 1
                    self.adjacency[j,i] = 1
                else:
                    self.adjacency[i,j] = 0
                    self.adjacency[j,i] = 0



        self.reward += -0.1

        self.t += 1

        if self.t >= 100:
            self.done = True

        if self.t % 100 == 0:
            #print('Target Location', (self.target_location[0],self.target_location[1]))
            #print("Adjacency Matrix", self.adjacency)

            print(self.reward)
            # plt.scatter(self.ant_locations[:,0],self.ant_locations[:,1])
            # plt.scatter(self.target_location[0],self.target_location[1])
            # plt.xlim([0, 16])
            # plt.ylim([0, 16])
            # plt.pause(1)
            # plt.close("all")

        return self.ant_locations, self.reward, self.adjacency, self.done




# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:30:12 2022

@author: jomor
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 21:09:47 2022

@author: jomor
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 22:19:26 2022

@author: jomor
"""

# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MjZ7gqR5_exd-1bRGz7KLx-0zH2ATEFJ
"""




# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:57:21 2022

@author: jomor
"""
import torch
class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.8, update_actor_interval=2, warmup=10000,
            n_actions=2, max_size=100000, layer1_size=400,
            layer2_size=300, layer3_size=200, batch_size=100, noise=0.1, agent_name = 'agent1'):
        self.gamma = gamma
        self.tau = tau
        self.max_action = [1]
        self.min_action = [0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.agent_name = agent_name
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, n_actions=n_actions,
                                  name='actor',agent_name = agent_name)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, layer3_size, n_actions=n_actions,
                                      name='critic_1',agent_name = agent_name)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, layer3_size, n_actions=n_actions,
                                      name='critic_2',agent_name = agent_name)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, n_actions=n_actions,
                                         name='target_actor',agent_name = agent_name)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                         layer2_size, layer3_size, n_actions=n_actions,
                                         name='target_critic_1',agent_name = agent_name)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                         layer2_size, layer3_size, n_actions=n_actions,
                                         name='target_critic_2',agent_name = agent_name)
        if self.agent_name == 'agent1' or self.agent_name == 'agent2' or self.agent_name == 'agent3' or self.agent_name == 'agent4':
            self.noise = noise
            #self.noise = noise
        else:
            self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, adjacency):

        ### eval mode ###
        #self.actor.eval()

        eps = 0.9
        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':

            if self.time_step < self.warmup:
                mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
                mu = T.abs(mu)
                #mu = mu / T.sum(mu)
                mu = mu.view(8,5)

                ### agent actions ###
                Z1 = mu[0] / T.sum(mu[0])
                Z2 = mu[1] / T.sum(mu[1])
                Z3 = mu[2] / T.sum(mu[2])
                Z4 = mu[3] / T.sum(mu[3])
                Z5 = mu[4] / T.sum(mu[4])
                Z6 = mu[5] / T.sum(mu[5])
                Z7 = mu[6] / T.sum(mu[6])
                Z8 = mu[7] / T.sum(mu[7])
                ### translate to environment vector ###
                mu1_ind = T.argmax(Z1)
                mu1_prime = Z1*0
                mu1_prime[mu1_ind] = 1

                mu2_ind = T.argmax(Z2)
                mu2_prime = Z2*0
                mu2_prime[mu2_ind] = 1

                mu3_ind = T.argmax(Z3)
                mu3_prime = Z3*0
                mu3_prime[mu3_ind] = 1

                mu4_ind = T.argmax(Z4)
                mu4_prime = Z4*0
                mu4_prime[mu4_ind] = 1

                mu5_ind = T.argmax(Z5)
                mu5_prime = Z5*0
                mu5_prime[mu5_ind] = 1

                mu6_ind = T.argmax(Z6)
                mu6_prime = Z6*0
                mu6_prime[mu6_ind] = 1

                mu7_ind = T.argmax(Z7)
                mu7_prime = Z7*0
                mu7_prime[mu7_ind] = 1

                mu8_ind = T.argmax(Z8)
                mu8_prime = Z8*0
                mu8_prime[mu8_ind] = 1



                mu1_prime = mu1_prime.cpu().detach().numpy()
                mu2_prime = mu2_prime.cpu().detach().numpy()
                mu3_prime = mu3_prime.cpu().detach().numpy()
                mu4_prime = mu4_prime.cpu().detach().numpy()
                mu5_prime = mu5_prime.cpu().detach().numpy()
                mu6_prime = mu6_prime.cpu().detach().numpy()
                mu7_prime = mu7_prime.cpu().detach().numpy()
                mu8_prime = mu8_prime.cpu().detach().numpy()

                Z1 = Z1.cpu().detach().numpy()
                Z2 = Z2.cpu().detach().numpy()
                Z3 = Z3.cpu().detach().numpy()
                Z4 = Z4.cpu().detach().numpy()

                Z5 = Z5.cpu().detach().numpy()
                Z6 = Z6.cpu().detach().numpy()
                Z7 = Z7.cpu().detach().numpy()
                Z8 = Z8.cpu().detach().numpy()

                self.time_step += 1

                return mu1_prime, mu2_prime, mu3_prime, mu4_prime, mu5_prime, mu6_prime, mu7_prime, mu8_prime, Z1, Z2, Z3, Z4 , Z5, Z6, Z7, Z8

            else:
                eps = 0.3
                state = 0
                #((4,32))
                #observation = observation.reshape((4*32))

                Z1 = T.tensor(observation[0:3], dtype=T.float).to(self.actor.device)
                Z2 = T.tensor(observation[3:6], dtype=T.float).to(self.actor.device)
                Z3 = T.tensor(observation[6:9], dtype=T.float).to(self.actor.device)
                Z4 = T.tensor(observation[9:12], dtype=T.float).to(self.actor.device)
                Z5 = T.tensor(observation[12:15], dtype=T.float).to(self.actor.device)
                Z6 = T.tensor(observation[15:18], dtype=T.float).to(self.actor.device)
                Z7 = T.tensor(observation[18:21], dtype=T.float).to(self.actor.device)
                Z8 = T.tensor(observation[21:24], dtype=T.float).to(self.actor.device)
                adjacency = T.tensor(observation[24:], dtype=T.float).to(self.actor.device)
                mu = self.actor.forward(state, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, adjacency)
                #print('mu',mu)

                mu = mu.view(8,5)

                ### agent actions ###
                Z1 = mu[0]
                Z2 = mu[1]
                Z3 = mu[2]
                Z4 = mu[3]

                Z5 = mu[4]
                Z6 = mu[5]
                Z7 = mu[6]
                Z8 = mu[7]


                self.time_step += 1
                ### translate to environment vector ###
                # Note that this is equivalent to what used to be called multinomial
                m = T.distributions.categorical.Categorical(Z1)
                action = m.sample()
                mu1_ind = action
                mu1_prime = Z1*0
                mu1_prime[mu1_ind] = 1

                m = T.distributions.categorical.Categorical(Z2)
                action = m.sample()
                mu2_ind = action
                mu2_prime = Z2*0
                mu2_prime[mu2_ind] = 1

                m = T.distributions.categorical.Categorical(Z3)
                action = m.sample()
                mu3_ind = action
                mu3_prime = Z3*0
                mu3_prime[mu3_ind] = 1

                m = T.distributions.categorical.Categorical(Z4)
                action = m.sample()
                mu4_ind = action
                mu4_prime = Z4*0
                mu4_prime[mu4_ind] = 1

                m = T.distributions.categorical.Categorical(Z5)
                action = m.sample()
                mu5_ind = action
                mu5_prime = Z5*0
                mu5_prime[mu5_ind] = 1

                m = T.distributions.categorical.Categorical(Z6)
                action = m.sample()
                mu6_ind = action
                mu6_prime = Z6*0
                mu6_prime[mu6_ind] = 1

                m = T.distributions.categorical.Categorical(Z7)
                action = m.sample()
                mu7_ind = action
                mu7_prime = Z7*0
                mu7_prime[mu7_ind] = 1

                m = T.distributions.categorical.Categorical(Z8)
                action = m.sample()
                mu8_ind = action
                mu8_prime = Z8*0
                mu8_prime[mu8_ind] = 1



                mu1 = mu1_prime.cpu().detach().numpy()
                mu2 = mu2_prime.cpu().detach().numpy()
                mu3 = mu3_prime.cpu().detach().numpy()
                mu4 = mu4_prime.cpu().detach().numpy()

                mu5 = mu5_prime.cpu().detach().numpy()
                mu6 = mu6_prime.cpu().detach().numpy()
                mu7 = mu7_prime.cpu().detach().numpy()
                mu8 = mu8_prime.cpu().detach().numpy()



                Z1 = Z1.cpu().detach().numpy()
                Z2 = Z2.cpu().detach().numpy()
                Z3 = Z3.cpu().detach().numpy()
                Z4 = Z4.cpu().detach().numpy()
                Z5 = Z5.cpu().detach().numpy()
                Z6 = Z6.cpu().detach().numpy()
                Z7 = Z7.cpu().detach().numpy()
                Z8 = Z8.cpu().detach().numpy()

                if np.random.uniform(0,1,1)[0] < eps:
                    mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
                    if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':
                        mu_ind = np.random.randint(2, size=8)

                        mu1 = np.abs(np.random.randn(5))
                        Z1 = mu1/np.sum(mu1)
                        mu1_ind = np.argmax(mu1)
                        mu1 = mu1*0
                        mu1[mu1_ind] = 1

                        mu2 = np.abs(np.random.randn(5))
                        Z2 = mu2/np.sum(mu2)
                        mu2_ind = np.argmax(mu2)
                        mu2 = mu2*0
                        mu2[mu2_ind] = 1

                        mu3 = np.abs(np.random.randn(5))
                        Z3 = mu3/np.sum(mu3)
                        mu3_ind = np.argmax(mu3)
                        mu3 = mu3*0
                        mu3[mu3_ind] = 1

                        mu4 = np.abs(np.random.randn(5))
                        Z4 = mu4/np.sum(mu4)
                        mu4_ind = np.argmax(mu4)
                        mu4 = mu4*0
                        mu4[mu4_ind] = 1

                        mu5 = np.abs(np.random.randn(5))
                        Z5 = mu5/np.sum(mu5)
                        mu5_ind = np.argmax(mu5)
                        mu5 = mu5*0
                        mu5[mu5_ind] = 1

                        mu6 = np.abs(np.random.randn(5))
                        Z6 = mu6/np.sum(mu6)
                        mu6_ind = np.argmax(mu6)
                        mu6 = mu6*0
                        mu6[mu6_ind] = 1

                        mu7 = np.abs(np.random.randn(5))
                        Z7 = mu7/np.sum(mu7)
                        mu7_ind = np.argmax(mu7)
                        mu7 = mu7*0
                        mu7[mu7_ind] = 1

                        mu8 = np.abs(np.random.randn(5))
                        Z8 = mu8/np.sum(mu8)
                        mu8_ind = np.argmax(mu8)
                        mu8= mu8*0
                        mu8[mu8_ind] = 1



                return mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8



        self.time_step += 1

        ### train mode ###
        #self.actor.train()

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        if self.agent_name == 'agent1k':
            Z1 = state_[:,0:3]
            Z2 = state_[:,3:6]
            Z3 = state_[:,6:9]
            Z4 = state_[:,9:12]
            Z5 = state_[:,12:15]
            Z6 = state_[:,15:18]
            Z7 = state_[:,18:21]
            Z8 = state_[:,21:24]
            A = state_[:,24:]
            target_actions = self.target_actor.forward(state_, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, A)

        Z1p = state[:,0:3]
        Z2p = state[:,3:6]
        Z3p = state[:,6:9]
        Z4p = state[:,9:12]
        Z5p = state[:,12:15]
        Z6p = state[:,15:18]
        Z7p = state[:,18:21]
        Z8p = state[:,21:24]
        Ap = state[:,24:]

        q1_ = self.target_critic_1.forward(state_, target_actions, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, A)
        q2_ = self.target_critic_2.forward(state_, target_actions, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, A)

        q1 = self.critic_1.forward(state, action, Z1p, Z2p, Z3p, Z4p, Z5p, Z6p, Z7p, Z8p, Ap)
        q2 = self.critic_2.forward(state, action, Z1p, Z2p, Z3p, Z4p, Z5p, Z6p, Z7p, Z8p, Ap)
        #print('q1', q1.shape)
        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)
        #print('c value',critic_value_.shape)
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1,1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # print('target', target.shape)
        # print('q1',q1.shape)
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
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state, Z1p, Z2p, Z3p, Z4p, Z5p, Z6p, Z7p, Z8p, Ap), Z1p, Z2p, Z3p, Z4p, Z5p, Z6p, Z7p, Z8p, Ap)
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        #print(actor_loss)

        self.update_network_parameters()

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
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

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

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:56:16 2022

@author: jomor
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions,
            name, chkpt_dir='tmp6/td3',agent_name='agent1'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.agent_name = agent_name
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir + '/' + agent_name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.agent_name = agent_name

        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':


            self.p1Layers = nn.ModuleList()
            self.p2Layers = nn.ModuleList()
            self.p3Layers = nn.ModuleList()
            self.p4Layers = nn.ModuleList()
            self.p5Layers = nn.ModuleList()
            self.p6Layers = nn.ModuleList()
            self.p7Layers = nn.ModuleList()
            self.p8Layers = nn.ModuleList()

            current_dim = 3
            hidden_dim = [256,128,64]
            self.ldim = len(hidden_dim)
            output_dim = 5

            for hdim in hidden_dim:
                self.p1Layers.append(nn.Linear(current_dim, hdim))
                self.p1Layers.append(nn.LayerNorm(hdim))
                #self.p1Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p1Layers.append(nn.Linear(current_dim*7, 7))

                self.p2Layers.append(nn.Linear(current_dim, hdim))
                self.p2Layers.append(nn.LayerNorm(hdim))
                #self.p2Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p2Layers.append(nn.Linear(current_dim*7, 7))

                self.p3Layers.append(nn.Linear(current_dim, hdim))
                self.p3Layers.append(nn.LayerNorm(hdim))
                #self.p3Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p3Layers.append(nn.Linear(current_dim*7, 7))

                self.p4Layers.append(nn.Linear(current_dim, hdim))
                self.p4Layers.append(nn.LayerNorm(hdim))
                #self.p4Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p4Layers.append(nn.Linear(current_dim*7, 7))

                self.p5Layers.append(nn.Linear(current_dim, hdim))
                self.p5Layers.append(nn.LayerNorm(hdim))
                #self.p5Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p5Layers.append(nn.Linear(current_dim*7, 7))

                self.p6Layers.append(nn.Linear(current_dim, hdim))
                self.p6Layers.append(nn.LayerNorm(hdim))
                #self.p6Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p6Layers.append(nn.Linear(current_dim*7, 7))

                self.p7Layers.append(nn.Linear(current_dim, hdim))
                self.p7Layers.append(nn.LayerNorm(hdim))
                #self.p7Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p7Layers.append(nn.Linear(current_dim*7, 7))

                self.p8Layers.append(nn.Linear(current_dim, hdim))
                self.p8Layers.append(nn.LayerNorm(hdim))
                #self.p8Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p8Layers.append(nn.Linear(current_dim*7, 7))

                current_dim = hdim

            # self.p1Layers.append(nn.Linear(current_dim, output_dim))
            # self.p2Layers.append(nn.Linear(current_dim, output_dim))
            # self.p3Layers.append(nn.Linear(current_dim, output_dim))
            # self.p4Layers.append(nn.Linear(current_dim, output_dim))
            # self.p5Layers.append(nn.Linear(current_dim, output_dim))
            # self.p6Layers.append(nn.Linear(current_dim, output_dim))
            # self.p7Layers.append(nn.Linear(current_dim, output_dim))
            # self.p8Layers.append(nn.Linear(current_dim, output_dim))

            self.q1 = nn.Linear(hdim*8 + 40, 256)
            self.b1 = nn.LayerNorm(256)
            self.q2 = nn.Linear(256, 128)
            self.b2 = nn.LayerNorm(128)
            self.q3 = nn.Linear(128, 64)
            self.b3 = nn.LayerNorm(64)
            self.q4 = nn.Linear(64, 1)


        elif self.agent_name == 'agent5':
            # dimensions: 24 + 2*32**2 + 4 + 16
            self.conv1 = nn.Conv2d(4, 64, 5)
            self.conv2 = nn.Conv2d(64, 32, 5)
            # an affine operation: y = Wx + b
            #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
            #self.rnn = nn.RNN(32 * 5 * 5 + 24 + 4 + self.n_actions, 256, 4, batch_first=True)
            #self.fc1 = nn.Linear(32 * 5 * 5 + 24 + 4, 256)  # 5*5 from image dimension
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.q1 = nn.Linear(64, 1)

        else:
            # I think this breaks if the env has a 2D state representation
            self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
            self.bn1 = nn.LayerNorm(self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            self.bn2 = nn.LayerNorm(self.fc2_dims)

            self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
            self.bn3 = nn.LayerNorm(self.fc3_dims)
            #self.q1 = nn.Linear(self.fc1_dims, 1)

            self.q1 = nn.Linear(self.fc1_dims, 1)



            if self.fc3_dims > 0:
                self.q1 = nn.Linear(self.fc3_dims, 1)

            elif self.fc2_dims > 0:
                self.q1 = nn.Linear(self.fc2_dims, 1)

            else:
                self.q1 = nn.Linear(self.fc1_dims, 1)


        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, A):
        # print(action.shape)
        # print(state.shape)
        # print(self.n_actions)
        # print(action)
        #print(self.agent_name)
        #print(state.shape)
        #print(action.shape)

        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':
            A = A.view(-1,8,8)
            # state = [Z1, Z2, Z3, Z4]
            ind = 0
            Z1 = Z1.view(-1,1,Z1.shape[-1])
            Z2 = Z2.view(-1,1,Z2.shape[-1])
            Z3 = Z3.view(-1,1,Z3.shape[-1])
            Z4 = Z4.view(-1,1,Z4.shape[-1])
            Z5 = Z5.view(-1,1,Z5.shape[-1])
            Z6 = Z6.view(-1,1,Z6.shape[-1])
            Z7 = Z7.view(-1,1,Z7.shape[-1])
            Z8 = Z8.view(-1,1,Z8.shape[-1])
            #print(len(self.p1Layers[:,-1]))
            action = action.view(-1,1,action.shape[-1])
            while ind < self.ldim*3:

                # Z1 = Z1.view(-1,1,Z1.shape[-1])
                # Z2 = Z2.view(-1,1,Z2.shape[-1])
                # Z3 = Z3.view(-1,1,Z3.shape[-1])
                # Z4 = Z4.view(-1,1,Z4.shape[-1])
                # Z5 = Z5.view(-1,1,Z5.shape[-1])
                # Z6 = Z6.view(-1,1,Z6.shape[-1])
                # Z7 = Z7.view(-1,1,Z7.shape[-1])
                # Z8 = Z8.view(-1,1,Z8.shape[-1])

                ### Importance Functions ###
                ### Z1 importance relations ###
                Zall1=T.cat([Z2, Z3, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a1SOFT = self.p1Layers[ind+2](Zall1)

                # a1Z2 = self.p1Layers[ind+2](Z2)
                # a1Z3 = self.p1Layers[ind+2](Z3)
                # a1Z4 = self.p1Layers[ind+2](Z4)
                # a1Z5 = self.p1Layers[ind+2](Z5)
                # a1Z6 = self.p1Layers[ind+2](Z6)
                # a1Z7 = self.p1Layers[ind+2](Z7)
                # a1Z8 = self.p1Layers[ind+2](Z8)
                # a1SOFT= T.cat([a1Z2, a1Z3, a1Z4, a1Z5, a1Z6, a1Z7, a1Z8],dim=-1)
                a1SOFT = F.softmax(a1SOFT, dim=2)

                ### Z1 importance relations ###
                Zall2=T.cat([Z1, Z3, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a2SOFT = self.p2Layers[ind+2](Zall2)
                # a2Z1 = self.p2Layers[ind+2](Z1)
                # a2Z3 = self.p2Layers[ind+2](Z3)
                # a2Z4 = self.p2Layers[ind+2](Z4)
                # a2Z5 = self.p2Layers[ind+2](Z5)
                # a2Z6 = self.p2Layers[ind+2](Z6)
                # a2Z7 = self.p2Layers[ind+2](Z7)
                # a2Z8 = self.p2Layers[ind+2](Z8)
                # a2SOFT=T.cat([a2Z1, a2Z3, a2Z4, a2Z5, a2Z6, a2Z7, a2Z8],dim=-1)
                a2SOFT = F.softmax(a2SOFT, dim=2)

                ### Z1 importance relations ###
                Zall3=T.cat([Z1, Z2, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a3SOFT = self.p3Layers[ind+2](Zall3)

                # a3Z2 = self.p3Layers[ind+2](Z2)
                # a3Z1 = self.p3Layers[ind+2](Z1)
                # a3Z4 = self.p3Layers[ind+2](Z4)
                # a3Z5 = self.p3Layers[ind+2](Z5)
                # a3Z6 = self.p3Layers[ind+2](Z6)
                # a3Z7 = self.p3Layers[ind+2](Z7)
                # a3Z8 = self.p3Layers[ind+2](Z8)
                # a3SOFT=T.cat([a3Z1, a3Z2, a3Z4, a3Z5, a3Z6, a3Z7, a3Z8],dim=-1)
                a3SOFT = F.softmax(a3SOFT, dim=2)

                ### Z1 importance relations ###
                Zall4=T.cat([Z1, Z2, Z3, Z5, Z6, Z7, Z8],dim=-1)
                a4SOFT = self.p4Layers[ind+2](Zall4)

                # a4Z2 = self.p4Layers[ind+2](Z2)
                # a4Z3 = self.p4Layers[ind+2](Z3)
                # a4Z1 = self.p4Layers[ind+2](Z1)
                # a4Z5 = self.p4Layers[ind+2](Z5)
                # a4Z6 = self.p4Layers[ind+2](Z6)
                # a4Z7 = self.p4Layers[ind+2](Z7)
                # a4Z8 = self.p4Layers[ind+2](Z8)
                # a4SOFT=T.cat([a4Z1, a4Z2, a4Z3, a4Z5, a4Z6, a4Z7, a4Z8],dim=-1)
                a4SOFT = F.softmax(a4SOFT, dim=2)

                ### Z1 importance relations ###
                Zall5=T.cat([Z1, Z2, Z3, Z4, Z6, Z7, Z8],dim=-1)
                a5SOFT = self.p5Layers[ind+2](Zall5)

                # a5Z2 = self.p5Layers[ind+2](Z2)
                # a5Z3 = self.p5Layers[ind+2](Z3)
                # a5Z4 = self.p5Layers[ind+2](Z4)
                # a5Z1 = self.p5Layers[ind+2](Z1)
                # a5Z6 = self.p5Layers[ind+2](Z6)
                # a5Z7 = self.p5Layers[ind+2](Z7)
                # a5Z8 = self.p5Layers[ind+2](Z8)
                # a5SOFT=T.cat([a5Z1, a5Z2, a5Z3, a5Z4, a5Z6, a5Z7, a5Z8],dim=-1)
                a5SOFT = F.softmax(a5SOFT, dim=2)

                ### Z1 importance relations ###
                Zall6=T.cat([Z1, Z2, Z3, Z4, Z5, Z7, Z8],dim=-1)
                a6SOFT = self.p6Layers[ind+2](Zall6)
                # a6Z2 = self.p6Layers[ind+2](Z2)
                # a6Z3 = self.p6Layers[ind+2](Z3)
                # a6Z4 = self.p6Layers[ind+2](Z4)
                # a6Z5 = self.p6Layers[ind+2](Z5)
                # a6Z1 = self.p6Layers[ind+2](Z1)
                # a6Z7 = self.p6Layers[ind+2](Z7)
                # a6Z8 = self.p6Layers[ind+2](Z8)
                # a6SOFT=T.cat([a6Z1, a6Z2, a6Z3, a6Z4, a6Z5, a6Z7, a6Z8],dim=-1)
                a6SOFT = F.softmax(a6SOFT, dim=2)

                ### Z1 importance relations ###
                Zall7=T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z8],dim=-1)
                a7SOFT = self.p7Layers[ind+2](Zall7)
                # a7Z2 = self.p7Layers[ind+2](Z2)
                # a7Z3 = self.p7Layers[ind+2](Z3)
                # a7Z4 = self.p7Layers[ind+2](Z4)
                # a7Z5 = self.p7Layers[ind+2](Z5)
                # a7Z6 = self.p7Layers[ind+2](Z6)
                # a7Z1 = self.p7Layers[ind+2](Z1)
                # a7Z8 = self.p7Layers[ind+2](Z8)
                # a7SOFT=T.cat([a7Z1, a7Z2, a7Z3, a7Z4, a7Z5, a7Z6, a7Z8],dim=-1)
                a7SOFT = F.softmax(a7SOFT, dim=2)

                ### Z1 importance relations ###
                Zall8=T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z7],dim=-1)
                a8SOFT = self.p8Layers[ind+2](Zall8)
                # a8Z3 = self.p8Layers[ind+2](Z3)
                # a8Z4 = self.p8Layers[ind+2](Z4)
                # a8Z5 = self.p8Layers[ind+2](Z5)
                # a8Z6 = self.p8Layers[ind+2](Z6)
                # a8Z7 = self.p8Layers[ind+2](Z7)
                # a8Z1 = self.p8Layers[ind+2](Z1)
                #a8T = a8Z2 + a8Z3 + a8Z4 + a8Z5 + a8Z6 + a8Z1 + a8Z
                # a8SOFT=T.cat([a8Z1, a8Z2, a8Z3, a8Z4, a8Z5, a8Z6, a8Z7],dim=-1)
                a8SOFT = F.softmax(a8SOFT, dim=2)
                # print( a8SOFT[:,0,0].shape)
                # print('Z1.shape',Z1.shape)
                # print('A',A[:,0,1].reshape(-1,1,1).shape)
                # ZZ = Z4*(A[:,0,3].reshape(-1,1,1))
                # print('ZZ',ZZ.shape)
                Z1_neigh_sum = a1SOFT[:,0,0].reshape(-1,1,1)*Z2*(A[:,0,1].reshape(-1,1,1)) + a1SOFT[:,0,1].reshape(-1,1,1)*Z3*(A[:,0,2].reshape(-1,1,1)) + a1SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,0,3].reshape(-1,1,1)) + a1SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,0].reshape(-1,1,1)) + a1SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,0].reshape(-1,1,1)) + a1SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,0].reshape(-1,1,1)) + a1SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,0].reshape(-1,1,1))
                Z2_neigh_sum = a2SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,1].reshape(-1,1,1)) + a2SOFT[:,0,1].reshape(-1,1,1)*Z3*(A[:,2,1].reshape(-1,1,1)) + a2SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,3,1].reshape(-1,1,1)) + a2SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,1].reshape(-1,1,1)) + a2SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,1].reshape(-1,1,1)) + a2SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,1].reshape(-1,1,1)) + a2SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,1].reshape(-1,1,1))
                Z3_neigh_sum = a3SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,2].reshape(-1,1,1)) + a3SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,2].reshape(-1,1,1)) + a3SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,3,2].reshape(-1,1,1)) + a3SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,2].reshape(-1,1,1)) + a3SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,2].reshape(-1,1,1)) + a3SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,2].reshape(-1,1,1)) + a3SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,2].reshape(-1,1,1))
                Z4_neigh_sum = a4SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,3].reshape(-1,1,1)) + a4SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,3].reshape(-1,1,1)) + a4SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,3].reshape(-1,1,1)) + a4SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,3].reshape(-1,1,1)) + a4SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,3].reshape(-1,1,1)) + a4SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,3].reshape(-1,1,1)) + a4SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,3].reshape(-1,1,1))
                Z5_neigh_sum = a5SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,4].reshape(-1,1,1)) + a5SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,4].reshape(-1,1,1)) + a5SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,4].reshape(-1,1,1)) + a5SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,4].reshape(-1,1,1)) + a5SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,4].reshape(-1,1,1)) + a5SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,4].reshape(-1,1,1)) + a5SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,4].reshape(-1,1,1))
                Z6_neigh_sum = a6SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,5].reshape(-1,1,1)) + a6SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,5].reshape(-1,1,1)) + a6SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,5].reshape(-1,1,1)) + a6SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,5].reshape(-1,1,1)) + a6SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,5].reshape(-1,1,1)) + a6SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,5].reshape(-1,1,1)) + a6SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,5].reshape(-1,1,1))
                Z7_neigh_sum = a7SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,6].reshape(-1,1,1)) + a7SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,6].reshape(-1,1,1)) + a7SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,6].reshape(-1,1,1)) + a7SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,6].reshape(-1,1,1)) + a7SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,6].reshape(-1,1,1)) + a7SOFT[:,0,5].reshape(-1,1,1)*Z6*(A[:,5,6].reshape(-1,1,1)) + a7SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,6].reshape(-1,1,1))
                Z8_neigh_sum = a8SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,7].reshape(-1,1,1)) + a8SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,7].reshape(-1,1,1)) + a8SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,7].reshape(-1,1,1)) + a8SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,7].reshape(-1,1,1)) + a8SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,7].reshape(-1,1,1)) + a8SOFT[:,0,5].reshape(-1,1,1)*Z6*(A[:,5,7].reshape(-1,1,1)) + a8SOFT[:,0,6].reshape(-1,1,1)*Z7*(A[:,6,7].reshape(-1,1,1))

                # print('A',A[:,0,1].reshape(-1,1,1).shape)
                # print('Z1 neigh shape1',Z1_neigh_sum.shape)
                #ZZ = Z2*(A[:,0,1].reshape(-1,1,1))
                #print(ZZ.shape)
                #print(Z2*(A[:,0,1].reshape(-1,1)))
                # Z1 = Z1.view(-1,1,Z1.shape[-1])
                # Z1_neigh_sum = Z1_neigh_sum.view(-1,1,Z1_neigh_sum.shape[-1])
                # #print('Z1 2',Z1.shape)
                # #print('Z1 neigh shape2',Z1_neigh_sum.shape)
                # Z2 = Z2.view(-1,1,Z2.shape[-1])
                # Z2_neigh_sum = Z2_neigh_sum.view(-1,1,Z2_neigh_sum.shape[-1])

                # Z3 = Z3.view(-1,1,Z3.shape[-1])
                # Z3_neigh_sum = Z3_neigh_sum.view(-1,1,Z3_neigh_sum.shape[-1])

                # Z4 = Z4.view(-1,1,Z4.shape[-1])
                # Z4_neigh_sum = Z4_neigh_sum.view(-1,1,Z4_neigh_sum.shape[-1])

                # Z5 = Z5.view(-1,1,Z5.shape[-1])
                # Z5_neigh_sum = Z5_neigh_sum.view(-1,1,Z5_neigh_sum.shape[-1])

                # Z6 = Z6.view(-1,1,Z6.shape[-1])
                # Z6_neigh_sum = Z6_neigh_sum.view(-1,1,Z6_neigh_sum.shape[-1])

                # Z7 = Z7.view(-1,1,Z7.shape[-1])
                # Z7_neigh_sum = Z7_neigh_sum.view(-1,1,Z7_neigh_sum.shape[-1])

                # Z8 = Z8.view(-1,1,Z8.shape[-1])
                # Z8_neigh_sum = Z8_neigh_sum.view(-1,1,Z8_neigh_sum.shape[-1])
                # print('Z1 neigh shape1',Z1_neigh_sum.shape)
                # print('Z8 after sum.shape',Z8.shape)
                #Z1, hn = self.p1Layers[ind+2](Z1, Z1_neigh_sum)
                Z1 = Z1 + Z1_neigh_sum
                # print('Z1 add.shape',Z1.shape)
                Z1 = F.relu(self.p1Layers[ind](Z1))
                Z1 = self.p1Layers[ind+1](Z1)
                #print('Z1 1',Z1.shape)
                # print('Z1 shape', Z1.shape)
                # print('Z1 neigh shape',Z1_neigh_sum.shape)

                #Z2, hn = self.p2Layers[ind+2](Z2, Z2_neigh_sum)
                Z2 = Z2 + Z2_neigh_sum
                Z2 = F.relu(self.p2Layers[ind](Z2))
                Z2 = self.p2Layers[ind+1](Z2)

                #Z3, hn = self.p3Layers[ind+2](Z3, Z3_neigh_sum)
                Z3 = Z3 + Z3_neigh_sum
                Z3 = F.relu(self.p3Layers[ind](Z3))
                Z3 = self.p3Layers[ind+1](Z3)

                #Z4, hn = self.p4Layers[ind+2](Z4, Z4_neigh_sum)
                Z4 = Z4 + Z4_neigh_sum
                Z4 = F.relu(self.p4Layers[ind](Z4))
                Z4 = self.p4Layers[ind+1](Z4)

                #Z5, hn = self.p5Layers[ind+2](Z5, Z5_neigh_sum)
                Z5 = Z5 + Z5_neigh_sum
                Z5 = F.relu(self.p5Layers[ind](Z5))
                Z5 = self.p5Layers[ind+1](Z5)

                #Z6, hn = self.p6Layers[ind+2](Z6, Z6_neigh_sum)
                Z6 = Z6 + Z6_neigh_sum
                Z6 = F.relu(self.p6Layers[ind](Z6))
                Z6 = self.p6Layers[ind+1](Z6)

                #Z7, hn = self.p7Layers[ind+2](Z7, Z7_neigh_sum)
                Z7 = Z7 + Z7_neigh_sum
                Z7 = F.relu(self.p7Layers[ind](Z7))
                Z7 = self.p7Layers[ind+1](Z7)

                #Z8, hn = self.p8Layers[ind+2](Z8, Z8_neigh_sum)
                Z8 = Z8 + Z8_neigh_sum
                Z8 = F.relu(self.p8Layers[ind](Z8))
                Z8 = self.p8Layers[ind+1](Z8)



                ind = ind + 3

            # Z1_neigh_sum = Z2*(A[:,0,1].reshape(-1,1,1)) + Z3*(A[:,0,2].reshape(-1,1,1)) + Z4*(A[:,0,3].reshape(-1,1,1)) + Z5*(A[:,4,0].reshape(-1,1,1)) + Z6*(A[:,5,0].reshape(-1,1,1)) + Z7*(A[:,6,0].reshape(-1,1,1)) +Z8*(A[:,7,0].reshape(-1,1,1))
            # Z2_neigh_sum = Z1*(A[:,0,1].reshape(-1,1,1)) + Z3*(A[:,2,1].reshape(-1,1,1)) + Z4*(A[:,3,1].reshape(-1,1,1)) + Z5*(A[:,4,1].reshape(-1,1,1)) + Z6*(A[:,5,1].reshape(-1,1,1)) + Z7*(A[:,6,1].reshape(-1,1,1)) + Z8*(A[:,7,1].reshape(-1,1,1))
            # Z3_neigh_sum = Z1*(A[:,0,2].reshape(-1,1,1)) + Z2*(A[:,1,2].reshape(-1,1,1)) + Z4*(A[:,3,2].reshape(-1,1,1)) + Z5*(A[:,4,2].reshape(-1,1,1)) + Z6*(A[:,5,2].reshape(-1,1,1)) + Z7*(A[:,6,2].reshape(-1,1,1)) + Z8*(A[:,7,2].reshape(-1,1,1))
            # Z4_neigh_sum = Z1*(A[:,0,3].reshape(-1,1,1)) + Z2*(A[:,1,3].reshape(-1,1,1)) + Z3*(A[:,2,3].reshape(-1,1,1)) + Z5*(A[:,4,3].reshape(-1,1,1)) + Z6*(A[:,5,3].reshape(-1,1,1)) + Z7*(A[:,6,3].reshape(-1,1,1)) + Z8*(A[:,7,3].reshape(-1,1,1))
            # Z5_neigh_sum = Z1*(A[:,0,4].reshape(-1,1,1)) + Z2*(A[:,1,4].reshape(-1,1,1))+ Z3*(A[:,2,4].reshape(-1,1,1)) + Z4*(A[:,3,4].reshape(-1,1,1)) + Z6*(A[:,5,4].reshape(-1,1,1)) + Z7*(A[:,6,4].reshape(-1,1,1)) + Z8*(A[:,7,4].reshape(-1,1,1))
            # Z6_neigh_sum = Z1*(A[:,0,5].reshape(-1,1,1)) + Z2*(A[:,1,5].reshape(-1,1,1)) + Z3*(A[:,2,5].reshape(-1,1,1)) + Z4*(A[:,3,5].reshape(-1,1,1)) + Z5*(A[:,4,5].reshape(-1,1,1)) + Z7*(A[:,6,5].reshape(-1,1,1)) + Z8*(A[:,7,5].reshape(-1,1,1))
            # Z7_neigh_sum = Z1*(A[:,0,6].reshape(-1,1,1)) + Z2*(A[:,1,6].reshape(-1,1,1)) + Z3*(A[:,2,6].reshape(-1,1,1)) + Z4*(A[:,3,6].reshape(-1,1,1)) + Z5*(A[:,4,6].reshape(-1,1,1)) + Z6*(A[:,5,6].reshape(-1,1,1)) + Z8*(A[:,7,6].reshape(-1,1,1))
            # Z8_neigh_sum = Z1*(A[:,0,7].reshape(-1,1,1)) + Z2*(A[:,1,7].reshape(-1,1,1)) + Z3*(A[:,2,7].reshape(-1,1,1)) + Z4*(A[:,3,7].reshape(-1,1,1)) + Z5*(A[:,4,7].reshape(-1,1,1)) + Z6*(A[:,5,7].reshape(-1,1,1)) + Z7*(A[:,6,7].reshape(-1,1,1))

            # print(Z1.shape)
            # print(Z1_neigh_sum.shape)
            # Z1_out = F.relu(self.p1Layers[-1](Z1))
            # Z2_out = F.relu(self.p2Layers[-1](Z2))
            # Z3_out = F.relu(self.p3Layers[-1](Z3))
            # Z4_out = F.relu(self.p4Layers[-1](Z4))
            #
            # Z5_out = F.relu(self.p5Layers[-1](Z5))
            # Z6_out = F.relu(self.p6Layers[-1](Z6))
            # Z7_out = F.relu(self.p7Layers[-1](Z7))
            # Z8_out = F.relu(self.p8Layers[-1](Z8))




            #F.gumbel_softmax(logits, tau=1, hard=False)
            # print(Z1.shape)
            # print(action.shape)
            Z_out = T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, action],dim=-1)
            #if self.time_step % 100 == 0:
              #print(Z_out.shape)
              #print('Zout',Z_out)
            #self.time_step += 1

            #print("Z_out shape",Z_out.shape)
            #T.cat([Z1_out, Z2_out, Z3_out, Z4_out],dim=1)

            #print('Z_out shape',Z_out.shape)
            # Z_out = Z_out.view(-1,Z_out.shape[-1])
            # # print('Z1out shape',Z1_out.shape)
            # print('Z_out shape',Z_out.shape)


            q1 = self.q1(Z_out)
            q1 = F.relu(q1)
            q1 = self.b1(q1)
            q1 = self.q2(q1)
            q1 = F.relu(q1)
            q1 = self.b2(q1)
            q1 = self.q3(q1)
            q1 = F.relu(q1)
            q1 = self.b3(q1)
            q1 = self.q4(q1)
            #print(q1.shape)


            return q1
        # if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':
        #     #state = state[:,0:128]
        #     state_action = T.cat([state,action],dim = 1)
        #     #print(state_action)
        #     q1_action_value = F.relu(self.fc1(T.cat([state,action],dim=1)))
        #     q1_action_value = self.bn1(q1_action_value)
        #     #q1_action_value = F.dropout(q1_action_value, p=0.1)
        #     #q1_action_value, hn = self.rnn1(q1_action_value)
        #
        #     q1_action_value = F.relu(self.fc2(q1_action_value))
        #     q1_action_value = self.bn2(q1_action_value)
        #     #q1_action_value = F.dropout(q1_action_value, p=0.1)
        #
        #     q1_action_value = self.fc3(q1_action_value)
        #     q1_action_value = F.relu(q1_action_value)
        #     q1_action_value = self.bn3(q1_action_value)
        #
        #     #q1_action_value, hn = self.rnn2(q1_action_value)
        #
        #
        #     #q1_action_value = F.dropout(q1_action_value, p=0.1)
        #     q1 = self.q1(q1_action_value)


        else:
            q1_action_value = self.fc1(T.cat([state, action], dim=1))
            q1_action_value = self.bn1(q1_action_value)
            q1_action_value = F.relu(q1_action_value)

            if self.fc2_dims > 0:
                #q1_action_value = F.dropout(q1_action_value, p=0.1)
                q1_action_value = self.fc2(q1_action_value)
                q1_action_value = self.bn2(q1_action_value)
                q1_action_value = F.relu(q1_action_value)

            if self.fc3_dims > 0:
                #q1_action_value = F.dropout(q1_action_value, p=0.1)
                q1_action_value = self.fc3(q1_action_value)
                q1_action_value = self.bn3(q1_action_value)
                q1_action_value = F.relu(q1_action_value)

            #q1_action_value = F.dropout(q1_action_value, p=0.1)

            q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file,map_location=torch.device('cpu')))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims,
            n_actions, name, chkpt_dir='tmp6/td3',agent_name='agent1'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        #self.hn = [0]
        self.agent_name = agent_name
        self.n_actions = n_actions
        self.name = name
        self.time_step = 0
        self.checkpoint_dir = chkpt_dir + '/' + agent_name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')


        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':


            self.p1Layers = nn.ModuleList()
            self.p2Layers = nn.ModuleList()
            self.p3Layers = nn.ModuleList()
            self.p4Layers = nn.ModuleList()
            self.p5Layers = nn.ModuleList()
            self.p6Layers = nn.ModuleList()
            self.p7Layers = nn.ModuleList()
            self.p8Layers = nn.ModuleList()

            current_dim = 3
            hidden_dim = [256, 128, 64]
            self.ldim = len(hidden_dim)
            output_dim1 = 64
            output_dim2 = 5

            for hdim in hidden_dim:
                self.p1Layers.append(nn.Linear(current_dim, hdim))
                self.p1Layers.append(nn.LayerNorm(hdim))
                #self.p1Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p1Layers.append(nn.Linear(current_dim*7, 7))

                self.p2Layers.append(nn.Linear(current_dim, hdim))
                self.p2Layers.append(nn.LayerNorm(hdim))
                #self.p2Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p2Layers.append(nn.Linear(current_dim*7, 7))

                self.p3Layers.append(nn.Linear(current_dim, hdim))
                self.p3Layers.append(nn.LayerNorm(hdim))
                #self.p3Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p3Layers.append(nn.Linear(current_dim*7, 7))

                self.p4Layers.append(nn.Linear(current_dim, hdim))
                self.p4Layers.append(nn.LayerNorm(hdim))
                #self.p4Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p4Layers.append(nn.Linear(current_dim*7, 7))


                self.p5Layers.append(nn.Linear(current_dim, hdim))
                self.p5Layers.append(nn.LayerNorm(hdim))
                #self.p5Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p5Layers.append(nn.Linear(current_dim*7, 7))

                self.p6Layers.append(nn.Linear(current_dim, hdim))
                self.p6Layers.append(nn.LayerNorm(hdim))
                #self.p6Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p6Layers.append(nn.Linear(current_dim*7, 7))

                self.p7Layers.append(nn.Linear(current_dim, hdim))
                self.p7Layers.append(nn.LayerNorm(hdim))
                #self.p7Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p7Layers.append(nn.Linear(current_dim*7, 7))

                self.p8Layers.append(nn.Linear(current_dim, hdim))
                self.p8Layers.append(nn.LayerNorm(hdim))
                #self.p8Layers.append(nn.GRU(current_dim, current_dim, 1, batch_first=True))
                self.p8Layers.append(nn.Linear(current_dim*7, 7))

                current_dim = hdim


            # self.p1Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p2Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p3Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p4Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p5Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p6Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p7Layers.append(nn.Linear(current_dim, output_dim1))
            # self.p8Layers.append(nn.Linear(current_dim, output_dim1))




            self.p1Layers.append(nn.Linear(current_dim, output_dim2))
            self.p2Layers.append(nn.Linear(current_dim, output_dim2))
            self.p3Layers.append(nn.Linear(current_dim, output_dim2))
            self.p4Layers.append(nn.Linear(current_dim, output_dim2))
            self.p5Layers.append(nn.Linear(current_dim, output_dim2))
            self.p6Layers.append(nn.Linear(current_dim, output_dim2))
            self.p7Layers.append(nn.Linear(current_dim, output_dim2))
            self.p8Layers.append(nn.Linear(current_dim, output_dim2))



        else:

          self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
          #self.rnn = nn.RNN(self.input_dims, self.fc1_dims, 4, batch_first=True)
          self.bn1 = nn.LayerNorm(self.fc1_dims)
          self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
          self.bn2 = nn.LayerNorm(self.fc2_dims)
          self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
          self.bn3 = nn.LayerNorm(self.fc3_dims)
          self.mu = nn.Linear(self.fc1_dims, self.n_actions)

          if self.fc2_dims > 0:
              self.mu = nn.Linear(self.fc2_dims, self.n_actions)
          if self.fc3_dims > 0:
              self.mu = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, A):

        if self.agent_name == 'agent1k' or self.agent_name == 'agent2k' or self.agent_name == 'agent3k' or self.agent_name == 'agent4k':
            A = A.view(-1,8,8)
            #state = [Z1, Z2, Z3, Z4]
            ind = 0
            Z1 = Z1.view(-1,1,Z1.shape[-1])
            Z2 = Z2.view(-1,1,Z2.shape[-1])
            Z3 = Z3.view(-1,1,Z3.shape[-1])
            Z4 = Z4.view(-1,1,Z4.shape[-1])
            Z5 = Z5.view(-1,1,Z5.shape[-1])
            Z6 = Z6.view(-1,1,Z6.shape[-1])
            Z7 = Z7.view(-1,1,Z7.shape[-1])
            Z8 = Z8.view(-1,1,Z8.shape[-1])
            #print(len(self.p1Layers[:,-1]))
            while ind < self.ldim*3:

                # Z1 = Z1.view(-1,1,Z1.shape[-1])
                # Z2 = Z2.view(-1,1,Z2.shape[-1])
                # Z3 = Z3.view(-1,1,Z3.shape[-1])
                # Z4 = Z4.view(-1,1,Z4.shape[-1])
                # Z5 = Z5.view(-1,1,Z5.shape[-1])
                # Z6 = Z6.view(-1,1,Z6.shape[-1])
                # Z7 = Z7.view(-1,1,Z7.shape[-1])
                # Z8 = Z8.view(-1,1,Z8.shape[-1])

                ### Importance Functions ###
                ### Z1 importance relations ###
                Zall1=T.cat([Z2, Z3, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a1SOFT = self.p1Layers[ind+2](Zall1)

                # a1Z2 = self.p1Layers[ind+2](Z2)
                # a1Z3 = self.p1Layers[ind+2](Z3)
                # a1Z4 = self.p1Layers[ind+2](Z4)
                # a1Z5 = self.p1Layers[ind+2](Z5)
                # a1Z6 = self.p1Layers[ind+2](Z6)
                # a1Z7 = self.p1Layers[ind+2](Z7)
                # a1Z8 = self.p1Layers[ind+2](Z8)
                # a1SOFT= T.cat([a1Z2, a1Z3, a1Z4, a1Z5, a1Z6, a1Z7, a1Z8],dim=-1)
                a1SOFT = F.softmax(a1SOFT, dim=2)

                ### Z1 importance relations ###
                Zall2=T.cat([Z1, Z3, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a2SOFT = self.p2Layers[ind+2](Zall2)
                # a2Z1 = self.p2Layers[ind+2](Z1)
                # a2Z3 = self.p2Layers[ind+2](Z3)
                # a2Z4 = self.p2Layers[ind+2](Z4)
                # a2Z5 = self.p2Layers[ind+2](Z5)
                # a2Z6 = self.p2Layers[ind+2](Z6)
                # a2Z7 = self.p2Layers[ind+2](Z7)
                # a2Z8 = self.p2Layers[ind+2](Z8)
                # a2SOFT=T.cat([a2Z1, a2Z3, a2Z4, a2Z5, a2Z6, a2Z7, a2Z8],dim=-1)
                a2SOFT = F.softmax(a2SOFT, dim=2)

                ### Z1 importance relations ###
                Zall3=T.cat([Z1, Z2, Z4, Z5, Z6, Z7, Z8],dim=-1)
                a3SOFT = self.p3Layers[ind+2](Zall3)

                # a3Z2 = self.p3Layers[ind+2](Z2)
                # a3Z1 = self.p3Layers[ind+2](Z1)
                # a3Z4 = self.p3Layers[ind+2](Z4)
                # a3Z5 = self.p3Layers[ind+2](Z5)
                # a3Z6 = self.p3Layers[ind+2](Z6)
                # a3Z7 = self.p3Layers[ind+2](Z7)
                # a3Z8 = self.p3Layers[ind+2](Z8)
                # a3SOFT=T.cat([a3Z1, a3Z2, a3Z4, a3Z5, a3Z6, a3Z7, a3Z8],dim=-1)
                a3SOFT = F.softmax(a3SOFT, dim=2)

                ### Z1 importance relations ###
                Zall4=T.cat([Z1, Z2, Z3, Z5, Z6, Z7, Z8],dim=-1)
                a4SOFT = self.p4Layers[ind+2](Zall4)

                # a4Z2 = self.p4Layers[ind+2](Z2)
                # a4Z3 = self.p4Layers[ind+2](Z3)
                # a4Z1 = self.p4Layers[ind+2](Z1)
                # a4Z5 = self.p4Layers[ind+2](Z5)
                # a4Z6 = self.p4Layers[ind+2](Z6)
                # a4Z7 = self.p4Layers[ind+2](Z7)
                # a4Z8 = self.p4Layers[ind+2](Z8)
                # a4SOFT=T.cat([a4Z1, a4Z2, a4Z3, a4Z5, a4Z6, a4Z7, a4Z8],dim=-1)
                a4SOFT = F.softmax(a4SOFT, dim=2)

                ### Z1 importance relations ###
                Zall5=T.cat([Z1, Z2, Z3, Z4, Z6, Z7, Z8],dim=-1)
                a5SOFT = self.p5Layers[ind+2](Zall5)

                # a5Z2 = self.p5Layers[ind+2](Z2)
                # a5Z3 = self.p5Layers[ind+2](Z3)
                # a5Z4 = self.p5Layers[ind+2](Z4)
                # a5Z1 = self.p5Layers[ind+2](Z1)
                # a5Z6 = self.p5Layers[ind+2](Z6)
                # a5Z7 = self.p5Layers[ind+2](Z7)
                # a5Z8 = self.p5Layers[ind+2](Z8)
                # a5SOFT=T.cat([a5Z1, a5Z2, a5Z3, a5Z4, a5Z6, a5Z7, a5Z8],dim=-1)
                a5SOFT = F.softmax(a5SOFT, dim=2)

                ### Z1 importance relations ###
                Zall6=T.cat([Z1, Z2, Z3, Z4, Z5, Z7, Z8],dim=-1)
                a6SOFT = self.p6Layers[ind+2](Zall6)
                # a6Z2 = self.p6Layers[ind+2](Z2)
                # a6Z3 = self.p6Layers[ind+2](Z3)
                # a6Z4 = self.p6Layers[ind+2](Z4)
                # a6Z5 = self.p6Layers[ind+2](Z5)
                # a6Z1 = self.p6Layers[ind+2](Z1)
                # a6Z7 = self.p6Layers[ind+2](Z7)
                # a6Z8 = self.p6Layers[ind+2](Z8)
                # a6SOFT=T.cat([a6Z1, a6Z2, a6Z3, a6Z4, a6Z5, a6Z7, a6Z8],dim=-1)
                a6SOFT = F.softmax(a6SOFT, dim=2)

                ### Z1 importance relations ###
                Zall7=T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z8],dim=-1)
                a7SOFT = self.p7Layers[ind+2](Zall7)
                # a7Z2 = self.p7Layers[ind+2](Z2)
                # a7Z3 = self.p7Layers[ind+2](Z3)
                # a7Z4 = self.p7Layers[ind+2](Z4)
                # a7Z5 = self.p7Layers[ind+2](Z5)
                # a7Z6 = self.p7Layers[ind+2](Z6)
                # a7Z1 = self.p7Layers[ind+2](Z1)
                # a7Z8 = self.p7Layers[ind+2](Z8)
                # a7SOFT=T.cat([a7Z1, a7Z2, a7Z3, a7Z4, a7Z5, a7Z6, a7Z8],dim=-1)
                a7SOFT = F.softmax(a7SOFT, dim=2)

                ### Z1 importance relations ###
                Zall8=T.cat([Z1, Z2, Z3, Z4, Z5, Z6, Z7],dim=-1)
                a8SOFT = self.p8Layers[ind+2](Zall8)
                # a8Z3 = self.p8Layers[ind+2](Z3)
                # a8Z4 = self.p8Layers[ind+2](Z4)
                # a8Z5 = self.p8Layers[ind+2](Z5)
                # a8Z6 = self.p8Layers[ind+2](Z6)
                # a8Z7 = self.p8Layers[ind+2](Z7)
                # a8Z1 = self.p8Layers[ind+2](Z1)
                #a8T = a8Z2 + a8Z3 + a8Z4 + a8Z5 + a8Z6 + a8Z1 + a8Z
                # a8SOFT=T.cat([a8Z1, a8Z2, a8Z3, a8Z4, a8Z5, a8Z6, a8Z7],dim=-1)
                a8SOFT = F.softmax(a8SOFT, dim=2)
                # print( a8SOFT[:,0,0].shape)
                # print('Z1.shape',Z1.shape)
                # print('A',A[:,0,1].reshape(-1,1,1).shape)
                # ZZ = Z4*(A[:,0,3].reshape(-1,1,1))
                # print('ZZ',ZZ.shape)
                Z1_neigh_sum = a1SOFT[:,0,0].reshape(-1,1,1)*Z2*(A[:,0,1].reshape(-1,1,1)) + a1SOFT[:,0,1].reshape(-1,1,1)*Z3*(A[:,0,2].reshape(-1,1,1)) + a1SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,0,3].reshape(-1,1,1)) + a1SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,0].reshape(-1,1,1)) + a1SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,0].reshape(-1,1,1)) + a1SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,0].reshape(-1,1,1)) + a1SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,0].reshape(-1,1,1))
                Z2_neigh_sum = a2SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,1].reshape(-1,1,1)) + a2SOFT[:,0,1].reshape(-1,1,1)*Z3*(A[:,2,1].reshape(-1,1,1)) + a2SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,3,1].reshape(-1,1,1)) + a2SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,1].reshape(-1,1,1)) + a2SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,1].reshape(-1,1,1)) + a2SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,1].reshape(-1,1,1)) + a2SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,1].reshape(-1,1,1))
                Z3_neigh_sum = a3SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,2].reshape(-1,1,1)) + a3SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,2].reshape(-1,1,1)) + a3SOFT[:,0,2].reshape(-1,1,1)*Z4*(A[:,3,2].reshape(-1,1,1)) + a3SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,2].reshape(-1,1,1)) + a3SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,2].reshape(-1,1,1)) + a3SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,2].reshape(-1,1,1)) + a3SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,2].reshape(-1,1,1))
                Z4_neigh_sum = a4SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,3].reshape(-1,1,1)) + a4SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,3].reshape(-1,1,1)) + a4SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,3].reshape(-1,1,1)) + a4SOFT[:,0,3].reshape(-1,1,1)*Z5*(A[:,4,3].reshape(-1,1,1)) + a4SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,3].reshape(-1,1,1)) + a4SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,3].reshape(-1,1,1)) + a4SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,3].reshape(-1,1,1))
                Z5_neigh_sum = a5SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,4].reshape(-1,1,1)) + a5SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,4].reshape(-1,1,1)) + a5SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,4].reshape(-1,1,1)) + a5SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,4].reshape(-1,1,1)) + a5SOFT[:,0,4].reshape(-1,1,1)*Z6*(A[:,5,4].reshape(-1,1,1)) + a5SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,4].reshape(-1,1,1)) + a5SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,4].reshape(-1,1,1))
                Z6_neigh_sum = a6SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,5].reshape(-1,1,1)) + a6SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,5].reshape(-1,1,1)) + a6SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,5].reshape(-1,1,1)) + a6SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,5].reshape(-1,1,1)) + a6SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,5].reshape(-1,1,1)) + a6SOFT[:,0,5].reshape(-1,1,1)*Z7*(A[:,6,5].reshape(-1,1,1)) + a6SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,5].reshape(-1,1,1))
                Z7_neigh_sum = a7SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,6].reshape(-1,1,1)) + a7SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,6].reshape(-1,1,1)) + a7SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,6].reshape(-1,1,1)) + a7SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,6].reshape(-1,1,1)) + a7SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,6].reshape(-1,1,1)) + a7SOFT[:,0,5].reshape(-1,1,1)*Z6*(A[:,5,6].reshape(-1,1,1)) + a7SOFT[:,0,6].reshape(-1,1,1)*Z8*(A[:,7,6].reshape(-1,1,1))
                Z8_neigh_sum = a8SOFT[:,0,0].reshape(-1,1,1)*Z1*(A[:,0,7].reshape(-1,1,1)) + a8SOFT[:,0,1].reshape(-1,1,1)*Z2*(A[:,1,7].reshape(-1,1,1)) + a8SOFT[:,0,2].reshape(-1,1,1)*Z3*(A[:,2,7].reshape(-1,1,1)) + a8SOFT[:,0,3].reshape(-1,1,1)*Z4*(A[:,3,7].reshape(-1,1,1)) + a8SOFT[:,0,4].reshape(-1,1,1)*Z5*(A[:,4,7].reshape(-1,1,1)) + a8SOFT[:,0,5].reshape(-1,1,1)*Z6*(A[:,5,7].reshape(-1,1,1)) + a8SOFT[:,0,6].reshape(-1,1,1)*Z7*(A[:,6,7].reshape(-1,1,1))

                # print('Z1 1',Z1.shape)
                # print('A',A[:,0,1].reshape(-1,1,1).shape)
                # print('Z1 neigh shape1',Z1_neigh_sum.shape)
                #ZZ = Z2*(A[:,0,1].reshape(-1,1,1))
                #print(ZZ.shape)
                #print(Z2*(A[:,0,1].reshape(-1,1)))
                # Z1 = Z1.view(-1,1,Z1.shape[-1])
                # Z1_neigh_sum = Z1_neigh_sum.view(-1,1,Z1_neigh_sum.shape[-1])
                # #print('Z1 2',Z1.shape)
                # #print('Z1 neigh shape2',Z1_neigh_sum.shape)
                # Z2 = Z2.view(-1,1,Z2.shape[-1])
                # Z2_neigh_sum = Z2_neigh_sum.view(-1,1,Z2_neigh_sum.shape[-1])

                # Z3 = Z3.view(-1,1,Z3.shape[-1])
                # Z3_neigh_sum = Z3_neigh_sum.view(-1,1,Z3_neigh_sum.shape[-1])

                # Z4 = Z4.view(-1,1,Z4.shape[-1])
                # Z4_neigh_sum = Z4_neigh_sum.view(-1,1,Z4_neigh_sum.shape[-1])

                # Z5 = Z5.view(-1,1,Z5.shape[-1])
                # Z5_neigh_sum = Z5_neigh_sum.view(-1,1,Z5_neigh_sum.shape[-1])

                # Z6 = Z6.view(-1,1,Z6.shape[-1])
                # Z6_neigh_sum = Z6_neigh_sum.view(-1,1,Z6_neigh_sum.shape[-1])

                # Z7 = Z7.view(-1,1,Z7.shape[-1])
                # Z7_neigh_sum = Z7_neigh_sum.view(-1,1,Z7_neigh_sum.shape[-1])

                # Z8 = Z8.view(-1,1,Z8.shape[-1])
                # Z8_neigh_sum = Z8_neigh_sum.view(-1,1,Z8_neigh_sum.shape[-1])


                #Z1, hn = self.p1Layers[ind+2](Z1, Z1_neigh_sum)
                Z1 = Z1 + Z1_neigh_sum
                Z1 = F.relu(self.p1Layers[ind](Z1))
                Z1 = self.p1Layers[ind+1](Z1)
                #print('Z1 1',Z1.shape)
                # print('Z1 shape', Z1.shape)
                # print('Z1 neigh shape',Z1_neigh_sum.shape)

                #Z2, hn = self.p2Layers[ind+2](Z2, Z2_neigh_sum)
                Z2 = Z2 + Z2_neigh_sum
                Z2 = F.relu(self.p2Layers[ind](Z2))
                Z2 = self.p2Layers[ind+1](Z2)

                #Z3, hn = self.p3Layers[ind+2](Z3, Z3_neigh_sum)
                Z3 = Z3 + Z3_neigh_sum
                Z3 = F.relu(self.p3Layers[ind](Z3))
                Z3 = self.p3Layers[ind+1](Z3)

                #Z4, hn = self.p4Layers[ind+2](Z4, Z4_neigh_sum)
                Z4 = Z4 + Z4_neigh_sum
                Z4 = F.relu(self.p4Layers[ind](Z4))
                Z4 = self.p4Layers[ind+1](Z4)

                #Z5, hn = self.p5Layers[ind+2](Z5, Z5_neigh_sum)
                Z5 = Z5 + Z5_neigh_sum
                Z5 = F.relu(self.p5Layers[ind](Z5))
                Z5 = self.p5Layers[ind+1](Z5)

                #Z6, hn = self.p6Layers[ind+2](Z6, Z6_neigh_sum)
                Z6 = Z6 + Z6_neigh_sum
                Z6 = F.relu(self.p6Layers[ind](Z6))
                Z6 = self.p6Layers[ind+1](Z6)

                #Z7, hn = self.p7Layers[ind+2](Z7, Z7_neigh_sum)
                Z7 = Z7 + Z7_neigh_sum
                Z7 = F.relu(self.p7Layers[ind](Z7))
                Z7 = self.p7Layers[ind+1](Z7)

                #Z8, hn = self.p8Layers[ind+2](Z8, Z8_neigh_sum)
                Z8 = Z8 + Z8_neigh_sum
                Z8 = F.relu(self.p8Layers[ind](Z8))
                Z8 = self.p8Layers[ind+1](Z8)



                ind = ind + 3

            # Z1_neigh_sum = Z2*(A[:,0,1].reshape(-1,1,1)) + Z3*(A[:,0,2].reshape(-1,1,1)) + Z4*(A[:,0,3].reshape(-1,1,1)) + Z5*(A[:,4,0].reshape(-1,1,1)) + Z6*(A[:,5,0].reshape(-1,1,1)) + Z7*(A[:,6,0].reshape(-1,1,1)) +Z8*(A[:,7,0].reshape(-1,1,1))
            # Z2_neigh_sum = Z1*(A[:,0,1].reshape(-1,1,1)) + Z3*(A[:,2,1].reshape(-1,1,1)) + Z4*(A[:,3,1].reshape(-1,1,1)) + Z5*(A[:,4,1].reshape(-1,1,1)) + Z6*(A[:,5,1].reshape(-1,1,1)) + Z7*(A[:,6,1].reshape(-1,1,1)) + Z8*(A[:,7,1].reshape(-1,1,1))
            # Z3_neigh_sum = Z1*(A[:,0,2].reshape(-1,1,1)) + Z2*(A[:,1,2].reshape(-1,1,1)) + Z4*(A[:,3,2].reshape(-1,1,1)) + Z5*(A[:,4,2].reshape(-1,1,1)) + Z6*(A[:,5,2].reshape(-1,1,1)) + Z7*(A[:,6,2].reshape(-1,1,1)) + Z8*(A[:,7,2].reshape(-1,1,1))
            # Z4_neigh_sum = Z1*(A[:,0,3].reshape(-1,1,1)) + Z2*(A[:,1,3].reshape(-1,1,1)) + Z3*(A[:,2,3].reshape(-1,1,1)) + Z5*(A[:,4,3].reshape(-1,1,1)) + Z6*(A[:,5,3].reshape(-1,1,1)) + Z7*(A[:,6,3].reshape(-1,1,1)) + Z8*(A[:,7,3].reshape(-1,1,1))
            # Z5_neigh_sum = Z1*(A[:,0,4].reshape(-1,1,1)) + Z2*(A[:,1,4].reshape(-1,1,1))+ Z3*(A[:,2,4].reshape(-1,1,1)) + Z4*(A[:,3,4].reshape(-1,1,1)) + Z6*(A[:,5,4].reshape(-1,1,1)) + Z7*(A[:,6,4].reshape(-1,1,1)) + Z8*(A[:,7,4].reshape(-1,1,1))
            # Z6_neigh_sum = Z1*(A[:,0,5].reshape(-1,1,1)) + Z2*(A[:,1,5].reshape(-1,1,1)) + Z3*(A[:,2,5].reshape(-1,1,1)) + Z4*(A[:,3,5].reshape(-1,1,1)) + Z5*(A[:,4,5].reshape(-1,1,1)) + Z7*(A[:,6,5].reshape(-1,1,1)) + Z8*(A[:,7,5].reshape(-1,1,1))
            # Z7_neigh_sum = Z1*(A[:,0,6].reshape(-1,1,1)) + Z2*(A[:,1,6].reshape(-1,1,1)) + Z3*(A[:,2,6].reshape(-1,1,1)) + Z4*(A[:,3,6].reshape(-1,1,1)) + Z5*(A[:,4,6].reshape(-1,1,1)) + Z6*(A[:,5,6].reshape(-1,1,1)) + Z8*(A[:,7,6].reshape(-1,1,1))
            # Z8_neigh_sum = Z1*(A[:,0,7].reshape(-1,1,1)) + Z2*(A[:,1,7].reshape(-1,1,1)) + Z3*(A[:,2,7].reshape(-1,1,1)) + Z4*(A[:,3,7].reshape(-1,1,1)) + Z5*(A[:,4,7].reshape(-1,1,1)) + Z6*(A[:,5,7].reshape(-1,1,1)) + Z7*(A[:,6,7].reshape(-1,1,1))

            # print(Z1.shape)
            # print(Z1_neigh_sum.shape)

            # Z1 = F.relu(self.p1Layers[-2](Z1))
            # Z2 = F.relu(self.p2Layers[-2](Z2))
            # Z3 = F.relu(self.p3Layers[-2](Z3))
            # Z4 = F.relu(self.p4Layers[-2](Z4))
            #
            # Z5 = F.relu(self.p5Layers[-2](Z5))
            # Z6 = F.relu(self.p6Layers[-2](Z6))
            # Z7 = F.relu(self.p7Layers[-2](Z7))
            # Z8 = F.relu(self.p8Layers[-2](Z8))

            Z1_out = F.gumbel_softmax(self.p1Layers[-1](Z1), tau=1, dim=2)
            Z2_out = F.gumbel_softmax(self.p2Layers[-1](Z2), tau=1, dim=2)
            Z3_out = F.gumbel_softmax(self.p3Layers[-1](Z3), tau=1, dim=2)
            Z4_out = F.gumbel_softmax(self.p4Layers[-1](Z4), tau=1, dim=2)

            Z5_out = F.gumbel_softmax(self.p5Layers[-1](Z5), tau=1, dim=2)
            Z6_out = F.gumbel_softmax(self.p6Layers[-1](Z6), tau=1, dim=2)
            Z7_out = F.gumbel_softmax(self.p7Layers[-1](Z7), tau=1, dim=2)
            Z8_out = F.gumbel_softmax(self.p8Layers[-1](Z8), tau=1, dim=2)

            # Z1_out = F.softmax(self.p1Layers[-1](Z1), dim=2)
            # Z2_out = F.softmax(self.p2Layers[-1](Z2), dim=2)
            # Z3_out = F.softmax(self.p3Layers[-1](Z3), dim=2)
            # Z4_out = F.softmax(self.p4Layers[-1](Z4), dim=2)

            # Z5_out = F.softmax(self.p5Layers[-1](Z5), dim=2)
            # Z6_out = F.softmax(self.p6Layers[-1](Z6), dim=2)
            # Z7_out = F.softmax(self.p7Layers[-1](Z7), dim=2)
            # Z8_out = F.softmax(self.p8Layers[-1](Z8), dim=2)


            #F.gumbel_softmax(logits, tau=1, hard=False)
            Z_out = T.cat([Z1_out, Z2_out, Z3_out, Z4_out, Z5_out, Z6_out, Z7_out, Z8_out],dim=-1)
            #if self.time_step % 100 == 0:
              #print(Z_out.shape)
              #print('Zout',Z_out)
            self.time_step += 1

            #print("Z_out shape",Z_out.shape)
            #T.cat([Z1_out, Z2_out, Z3_out, Z4_out],dim=1)

            #print('Z_out shape',Z_out.shape)
            #Z_out = Z_out.view(-1,Z_out.shape[-1])
            # print('Z1out shape',Z1_out.shape)
            # print('Z_out shape',Z_out.shape)
            return Z_out


        elif self.agent_name == 'agent5':
            # dimensions: 8 + 2*32**2 + 2 + 16
            ### reshape tensor ###
            #state = state.reshape([ 64, 1, 32, 32])
            state = state.view(-1, 24 + 4*32**2 + 4)
            #print(state.shape,state.shape[0])
            state_conv = state[:,24:24 + 4*32**2]
            #print(state_conv.shape)
            state_conv = state_conv.view(-1, 4, 32, 32)
            # Max pooling over a (2, 2) window
            prob = F.max_pool2d(F.relu(self.conv1(state_conv)), (2, 2))
            # If the size is a square, you can specify with a single number
            prob = F.max_pool2d(F.relu(self.conv2(prob)), 2)
            prob = torch.flatten(prob, 1) # flatten all dimensions except the batch dimension
            #F.relu(self.fc1(T.cat([state_conv,state[:,0:8],state[:,8 + 2*32**2:]],dim=1)))
            #prob = F.relu(self.fc1(T.cat([prob,state[:,0:24],state[:, 24 + 4*32**2:]],dim=1)))
            #h0 = torch.zeros(2, prob.size(0), 256).requires_grad_()
            prob, hn = self.rnn(T.cat([prob,state[:,0:24],state[:, 24 + 4*32**2:]],dim=1))
            prob = F.relu(prob)
            # if len(self.hn) <= 2:
            #   prob, hn = self.rnn(T.cat([prob,state[:,0:24],state[:, 24 + 4*32**2:]],dim=1))
            #   prob = F.relu(prob)
            # else:
            #   #prob, hn = F.relu(self.rnn(T.cat([prob,state[:,0:24],state[:, 24 + 4*32**2:]],dim=1),self.hn))
            #   prob, hn = self.rnn(T.cat([prob,state[:,0:24],state[:, 24 + 4*32**2:]],dim=1),self.hn)
            #   prob = F.relu(prob)
            self.hn = hn
            #prob = F.dropout(prob, p=0.1)
            prob = F.relu(self.fc2(prob))
            #prob = F.dropout(prob, p=0.1)
            prob = self.fc3(prob)
            prob = F.relu(prob)
            prob = F.dropout(prob, p=0.1)

        else:
          #prob = self.fc1(state)
          #state = state.view(-1, 1, self.input_dims)
          #print(state.shape)
          prob = self.fc1(state)
          prob = F.relu(prob)
          prob = self.bn1(prob)
          prob = F.relu(prob)

          if self.fc2_dims > 0:
              #prob = F.dropout(prob, p=0.1)
              prob = self.fc2(prob)
              prob = self.bn2(prob)
              prob = F.relu(prob)

          if self.fc3_dims > 0:
              #prob = F.dropout(prob, p=0.1)
              prob = self.fc3(prob)
              prob = self.bn3(prob)
              prob = F.relu(prob)
          #prob = F.dropout(prob, p=0.1)
        if self.agent_name == 'agent1' or self.agent_name == 'agent2' or self.agent_name == 'agent3' or self.agent_name == 'agent4' or self.agent_name == 'agent5':
            prob = T.sigmoid(self.mu(prob)) # if action is > +/- 1 then multiply by max action
        else:
            prob = T.sigmoid(self.mu(prob))

        return prob

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file,map_location=torch.device('cpu')))

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:53:35 2022

@author: jomor
"""

import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones



# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:58:07 2022

@author: jomor
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    #plt.savefig(figure_file)

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:55:20 2022

@author: jomor
"""

import gym
import numpy as np

if __name__ == '__main__':
    #env = gym.make('LunarLanderContinuous-v2')
    # agent1 = Agent(alpha=0.000001, beta=0.000001,
    #             input_dims=3, tau=0.1,
    #             env=0, batch_size=256, layer1_size=512, layer2_size=256, layer3_size=64,
    #             n_actions = 35, agent_name = 'agent1')

    # agent2 = Agent(alpha=0.000001, beta=0.000001,
    #             input_dims=3, tau=0.1,
    #             env=0, batch_size=256, layer1_size=512, layer2_size=256, layer3_size=64,
    #             n_actions=35, agent_name = 'agent2')

    # agent3 = Agent(alpha=0.000001, beta=0.000001,
    #             input_dims=3, tau=0.1,
    #             env=0, batch_size=256, layer1_size=512, layer2_size=256, layer3_size=64,
    #             n_actions = 35, agent_name = 'agent3')

    # agent4 = Agent(alpha=0.000001, beta=0.000001,
    #             input_dims=3, tau=0.01,
    #             env=0, batch_size=256, layer1_size=512, layer2_size=256, layer3_size=64,
    #             n_actions=35, agent_name = 'agent4')

    agent1k = Agent(alpha=0.000001, beta=0.000001,
                input_dims=8*3+8*8, tau=0.0001,
                env=0, batch_size=128, layer1_size=512, layer2_size=256, layer3_size=64,
                n_actions=40, agent_name = 'agent1k')


    # agent5 = Agent(alpha=0.0001, beta=0.0001,
    #             input_dims=24 + 4*32**2 + 4, tau=0.01,
    #             env=env, batch_size=32, layer1_size=256, layer2_size=128,
    #             n_actions=20, agent_name = 'agent5')



    #print(env.action_space.shape[0])


    n_games = 2500
    #filename = 'Walker2d_' + str(n_games) + '_2.png'
    #figure_file = 'plots/' + filename

    best_score = 0
    score_history = []

    #agent.load_models()
    # agent1.load_models()
    # agent2.load_models()
    # agent3.load_models()
    # agent4.load_models()
    agent1k.load_models()
    # agent2k.load_models()
    # agent3k.load_models()
    # agent4k.load_models()
    # agent5.load_models()
    observation1 = np.zeros(3)
    observation2 = np.zeros(3)
    observation3 = np.zeros(3)
    observation4 = np.zeros(3)

    observation5 = np.zeros(3)
    observation6 = np.zeros(3)
    observation7 = np.zeros(3)
    observation8 = np.zeros(3)


    env = ant_hill()
    env.init()
    ant_locations, adjacency = env.reset()

    for i in range(n_games):
        ### initialize and get ant locations ###
        ### each ant agent take its location and hill_staus as observation
        ant_locations, adjacency = env.reset()

        observation1 = ant_locations[0]
        observation2 = ant_locations[1]
        observation3 = ant_locations[2]
        observation4 = ant_locations[3]

        observation5 = ant_locations[4]
        observation6 = ant_locations[5]
        observation7 = ant_locations[6]
        observation8 = ant_locations[7]



        done = False
        score = 0
        turn = 0
        while not done:

            # ### agent 1 action ###
            # action1 = agent1.choose_action(observation1,adjacency)

            # ### agent 2 action ###
            # action2 = agent2.choose_action(observation2,adjacency)

            # ### agent 1 action ###
            # action3 = agent3.choose_action(observation3,adjacency)

            # ### agent 2 action ###
            # action4 = agent4.choose_action(observation4,adjacency)

            ### Put actions together ###
            action_env = np.zeros((8,5))

            # action_env[0,:] = action1[:]
            # action_env[1,:] = action2[:]
            # action_env[2,:] = action3[:]
            # action_env[3,:] = action4[:]


            #if turn > 0:
                ###
                ### note: an agent can send message to three nodes
                ### the first non-encoding index is index 16
                ### this means the agent sends to the lowest
                ### numbered node not including itself
                ### 17 goes to second lowest
                ### 18 goes to third
                ###


                ### if turn>0, agent 1k, 2k, 3k and 4k can learn ###
                # ### new observation1k ###
                # observation1k_ = np.zeros(8*3 + 8*8)
                # observation1k_[0:3] = observation1
                # observation1k_[3:6] = observation2
                # observation1k_[6:9] = observation3
                # observation1k_[9:12] = observation4
                # observation1k_[12:15] = observation5
                # observation1k_[15:18] = observation6
                # observation1k_[18:21] = observation7
                # observation1k_[21:24] = observation8
                # observation1k_[24:] = adjacency.reshape(8*8)
                #
                # action1k = np.zeros(8*5)
                # action1k = action_env.reshape(8*5)
                #
                # action_learn = np.zeros((8,5))
                # # print(action1k_raw)
                # action_learn[0,:] = np.round(action1k_raw[4])
                # action_learn[1,:] = np.round(action1k_raw[5])
                # action_learn[2,:] = np.round(action1k_raw[6])
                # action_learn[3,:] = np.round(action1k_raw[7])
                # action_learn[4,:] = np.round(action1k_raw[8])
                # action_learn[5,:] = np.round(action1k_raw[9])
                # action_learn[6,:] = np.round(action1k_raw[10])
                # action_learn[7,:] = np.round(action1k_raw[11])
                #
                #
                # #observation1k_ = observation1k_.reshape(4*32)
                # #observation1k = observation1k.reshape(4*32)
                # action_learn = action_learn.reshape(8*5)
                #
                # ### agent1k learn ###
                # agent1k.remember(observation1k, action_learn, reward, observation1k_, done)
                # agent1k.learn()



            ### new observation1k ###
            observation1k = np.zeros(8*3 + 8*8)
            observation1k[0:3] = observation1
            observation1k[3:6] = observation2
            observation1k[6:9] = observation3
            observation1k[9:12] = observation4
            observation1k[12:15] = observation5
            observation1k[15:18] = observation6
            observation1k[18:21] = observation7
            observation1k[21:24] = observation8
            observation1k[24:] = adjacency.reshape(8*8)

            ### agent1k and agent2k, agent3k, and agent4k take action ###
            ### agent1k action ###
            action1k_raw = agent1k.choose_action(observation1k,adjacency)
            # act_ind = np.argmax(action1k_raw)
            # action1k[act_ind] = 1

            #print('action1k[1]',action1k_raw[1])



            ### construct action ###
            action_env = np.zeros((8,5))
            # print(action1k_raw)
            action_env[0,:] = np.round(action1k_raw[0])
            action_env[1,:] = np.round(action1k_raw[1])
            action_env[2,:] = np.round(action1k_raw[2])
            action_env[3,:] = np.round(action1k_raw[3])
            action_env[4,:] = np.round(action1k_raw[4])
            action_env[5,:] = np.round(action1k_raw[5])
            action_env[6,:] = np.round(action1k_raw[6])
            action_env[7,:] = np.round(action1k_raw[7])

            #action_env = np.array([action3[],action4[0]])

            ### get new ant_locations ###
            ant_locations, reward, adjacency, done = env.step(action_env)
            #print('reward',reward)
            #print(ant_locations)


            observation1_ = ant_locations[0]
            observation2_ = ant_locations[1]
            observation3_ = ant_locations[2]
            observation4_ = ant_locations[3]

            observation5_ = ant_locations[4]
            observation6_ = ant_locations[5]
            observation7_ = ant_locations[6]
            observation8_ = ant_locations[7]

            ### agent1 and agent2 has information to learn ###
            # observation1_ = np.zeros(3)
            # observation2_ = np.zeros(3)
            # observation3_ = np.zeros(3)
            # observation4_ = np.zeros(3)
            #
            # observation5_ = np.zeros(3)
            # observation6_ = np.zeros(3)
            # observation7_ = np.zeros(3)
            # observation8_ = np.zeros(3)



            ### new observation1k ###
            observation1k_ = np.zeros(8*3 + 8*8)
            observation1k_[0:3] = observation1_
            observation1k_[3:6] = observation2_
            observation1k_[6:9] = observation3_
            observation1k_[9:12] = observation4_
            observation1k_[12:15] = observation5_
            observation1k_[15:18] = observation6_
            observation1k_[18:21] = observation7_
            observation1k_[21:24] = observation8_
            observation1k_[24:] = adjacency.reshape(8*8)

            action1k = np.zeros(8*5)
            action1k = action_env.reshape(8*5)

            action_learn = np.zeros((8,5))
            # print(action1k_raw)
            action_learn[0,:] = action1k_raw[8]
            action_learn[1,:] = action1k_raw[9]
            action_learn[2,:] = action1k_raw[10]
            action_learn[3,:] = action1k_raw[11]
            action_learn[4,:] = action1k_raw[12]
            action_learn[5,:] = action1k_raw[13]
            action_learn[6,:] = action1k_raw[14]
            action_learn[7,:] = action1k_raw[15]


            #observation1k_ = observation1k_.reshape(4*32)
            #observation1k = observation1k.reshape(4*32)
            action_learn = action_learn.reshape(8*5)

            ### agent1k learn ###
            agent1k.remember(observation1k, action_learn, reward, observation1k_, done)
            agent1k.learn()


            ### set agent1-4 new observation ###
            observation1 = observation1_
            observation2 = observation2_
            observation3 = observation3_
            observation4 = observation4_

            observation5 = observation5_
            observation6 = observation6_
            observation7 = observation7_
            observation8 = observation8_


            score += reward
            turn += 1

        if turn%100 == 0:
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            print('action_env',action_env)
            print('action1k_raw',action1k_raw)
            if i%10 == 0:
                best_score = avg_score
                # agent1.save_models()
                # agent2.save_models()
                # agent3.save_models()
                # agent4.save_models()
                agent1k.save_models()
                # agent2k.save_models()
                # agent3k.save_models()
                # agent4k.save_models()
                # agent5.save_models()
            print('episode ', i, 'score %.2f' % score,
                    'trailing 100 games avg %.3f' % avg_score)
# agent1.save_models()
# agent2.save_models()
# agent3.save_models()
# agent4.save_models()
agent1k.save_models()
# agent2k.save_models()
# agent3k.save_models()
# agent4k.save_models()
# # agent5.save_models()
