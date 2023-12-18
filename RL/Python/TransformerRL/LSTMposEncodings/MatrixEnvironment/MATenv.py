# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 06:30:19 2023

@author: jomor
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import h5py



class MatrixAttentionTask(gym.Env):

    def __init__(self):
        super(MatrixAttentionTask, self).__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        self.p_cue = 1 / 3.0
        self.t_cue = 1
        self.del_cue = 2
        self.t_stim = self.t_cue + self.del_cue + 1
        self.del_stim = 2
        self.T_end = self.t_stim + self.del_stim
        self.target_value = 0.4
        self.fig, self.ax = None, None

    def reset(self):
        self.t = 0

        rand_num = np.random.random()
        if rand_num < self.p_cue:
            self.cue = 'S1'
        elif rand_num < 2 * self.p_cue:
            self.cue = 'S4'
        else:
            self.cue = 'both'

        rand_val = np.random.random()
        if rand_val < 1 / 5:
            self.target = 'none'
        else:
            if self.cue == 'S1':
                if np.random.random() < 6 / 12:
                    self.target = 'S1'
                elif np.random.random() < 1:
                    self.target = 'S2'
                elif np.random.random() < 0:
                    self.target = 'S3'
                elif np.random.random() < 0:
                    self.target = 'S4'

            elif self.cue == 'S4':
                if np.random.random() < 6 / 12:
                    self.target = 'S3'
                elif np.random.random() < 1:
                    self.target = 'S4'
                elif np.random.random() < 0:
                    self.target = 'S1'
                elif np.random.random() <= 0:
                    self.target = 'S2'

            elif self.cue == 'both':
                rand_val = np.random.random()
                if rand_val < 1 / 5:
                    self.target = 'none'
                elif np.random.random() < 2 / 5:
                    self.target = 'S1'
                elif np.random.random() < 3 / 5:
                    self.target = 'S2'
                elif np.random.random() < 4 / 5:
                    self.target = 'S3'
                elif np.random.random() <= 1:
                    self.target = 'S4'

        self.num_target_features = np.random.randint(1, 5)
        self.target_indices = np.random.choice(4, self.num_target_features, replace=False)

        # print(self.cue, self.target, self.num_target_features, self.target_indices)

        # print('WHY',self._next_observation().shape)

        return self._next_observation()

    def _next_observation(self):
        S1 = np.zeros((2, 2))
        S2 = np.zeros((2, 2))
        S3 = np.zeros((2, 2))
        S4 = np.zeros((2, 2))

        if self.t_cue <= self.t < self.t_cue + self.del_cue:
            if self.cue in ['S1', 'both']:
                S1.fill(1)
            if self.cue in ['S4', 'both']:
                S4.fill(1)

        elif self.t_stim <= self.t <= self.T_end:
            S1.fill(0.5)
            S2.fill(0.5)
            S3.fill(0.5)
            S4.fill(0.5)

            if self.target == 'S1':
                for index in self.target_indices:
                    i, j = divmod(index, 2)
                    S1[i, j] += self.target_value

            elif self.target == 'S2':
                for index in self.target_indices:
                    i, j = divmod(index, 2)
                    S2[i, j] += self.target_value

            elif self.target == 'S3':
                for index in self.target_indices:
                    i, j = divmod(index, 2)
                    S3[i, j] += self.target_value

            elif self.target == 'S4':
                for index in self.target_indices:
                    i, j = divmod(index, 2)
                    S4[i, j] += self.target_value

            S1 += np.random.normal(0, 0.1, (2, 2))
            S2 += np.random.normal(0, 0.1, (2, 2))
            S3 += np.random.normal(0, 0.1, (2, 2))
            S4 += np.random.normal(0, 0.1, (2, 2))

        return np.concatenate([S1.ravel(), S2.ravel(), S3.ravel(), S4.ravel()])

    def step(self, action):

        reward = 0
        done = False

        if action == 1:
            # print('action NOT ZERO!',action)
            done = True
            if self.t < self.t_stim:
                reward = -1
            elif self.target == 'S1' or self.target == 'S2':
                reward = 1
            elif self.target == 'none' or self.target == 'S3' or self.target == 'S4':  # wrong classification during ISI
                reward = 0
            else:
                'whoops! Shouldnt be here!'

        elif action == 2:
            # print('action NOT ZERO!',action)
            done = True
            if self.t < self.t_stim:
                reward = -1
            elif self.target == 'S3' or self.target == 'S4':
                reward = 1
            elif self.target == 'none' or self.target == 'S1' or self.target == 'S2':  # wrong classification during ISI
                reward = 0
            else:
                'whoops! Shouldnt be here!'


        self.t += 1
        if self.t >= self.T_end:
            done = True
            if self.target == 'none' and action == 0:
                reward = 1
            else:
                reward = 0

        return self._next_observation(), reward, done, {}

    def close(self):
        pass

    def render(self, mode="human"):
        obs = self._next_observation()
        S1 = obs[:4].reshape(2, 2)
        S2 = obs[4:8].reshape(2, 2)
        S3 = obs[8:12].reshape(2, 2)
        S4 = obs[12:16].reshape(2, 2)

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(2, 2)
            self.ax[0,0].imshow(S1, cmap='viridis', vmin=0, vmax=1)
            self.ax[0,0].set_title("S1")
            self.ax[0,1].imshow(S2, cmap='viridis', vmin=0, vmax=1)
            self.ax[0,1].set_title("S2")
            self.ax[1, 0].imshow(S3, cmap='viridis', vmin=0, vmax=1)
            self.ax[1, 0].set_title("S3")
            self.ax[1, 1].imshow(S4, cmap='viridis', vmin=0, vmax=1)
            self.ax[1, 1].set_title("S4")
        else:
            self.ax[0,0].images[0].set_array(S1)
            self.ax[0,1].images[0].set_array(S2)
            self.ax[1, 0].images[0].set_array(S3)
            self.ax[1, 1].images[0].set_array(S4)

        plt.pause(0.5)
# # #
# # # # # #
# env = MatrixAttentionTask()
# n_samples = 10**6
#
# labels_target = np.zeros((n_samples,env.T_end+1,1))
# labels_cue = np.zeros((n_samples,env.T_end+1,1))
# labels_time = np.zeros((n_samples,env.T_end+1,1))
# labels_ATTprop = np.zeros((n_samples,env.T_end+1,1))
#
#
# samples = np.zeros((n_samples,env.T_end+1,16))
#
# eps1 = 0.2
# eps2 = 0.5
#
# # ### full sample games ###
# for i in range(n_samples):
#
#     ### Determine if an all or nothing selection happened ###
#     eps = np.random.random()
#
#     ### sample array ###
#     sample = np.zeros((env.T_end+1,16))
#     label_target = np.zeros((env.T_end+1,1))
#     label_cue = np.zeros((env.T_end+1,1))
#     label_time = np.zeros((env.T_end+1,1))
#     label_ATTprop = np.zeros((env.T_end+1,1))
#
#
#     observation = env.reset()
#     done = False
#     score = 0
#     alpha1 = 0.5
#     alpha2 = 0.5
#
#     sample[env.t,:] = observation[:]
#     if env.target == 'none':
#         label_target[:, 0] = 0
#     elif env.target == 'S1':
#         label_target[:, 0] = 1
#     elif env.target == 'S2':
#         label_target[:, 0] = 1
#     elif env.target == 'S3':
#         label_target[:, 0] = 2
#     elif env.target == 'S4':
#         label_target[:, 0] = 2
#
#     if env.cue == 'both':
#         label_cue[:, 0] = 0
#     elif env.cue == 'S1':
#         label_cue[:, 0] = 1
#     elif env.cue == 'S2':
#         label_cue[:, 0] = 2
#     elif env.cue == 'S3':
#         label_cue[:, 0] = 3
#     elif env.cue == 'S4':
#         label_cue[:, 0] = 4
#
#     label_time[env.t] = env.t
#
#     label_ATTprop[env.t] = np.sum(sample[0:env.t+1,4])/(env.t+1)
#
#     last_attended = -1
#     while not done:
#         action=0
#         # print(env.t, env.cue, env.t_stim,env.target)
#
#         observation_, reward, done, _ = env.step(action)
#
#         sample[env.t, :] = observation_[0:16]
#
#
#         label_time[env.t] = env.t
#
#         label_ATTprop[env.t] = np.sum(sample[0:env.t+1,4])/(env.t+1)
#
#
#     samples[i] = sample
#     labels_target[i] = label_target
#     labels_cue[i] = label_cue
#     labels_time[i] = label_time
#     labels_ATTprop[i] = label_ATTprop
#
#     if i % 10**4 == 0:
#         print(i)
#         print(label_target)
#
#
# h5f = h5py.File('train_data_shortTrial.h5', 'w')
# h5f.create_dataset('full_samples', data=samples)
# h5f.create_dataset('labels_target', data=labels_target)
# h5f.create_dataset('labels_cue', data=labels_cue)
# h5f.create_dataset('labels_time', data=labels_time)
# h5f.create_dataset('labels_ATTprop', data=labels_ATTprop)
# h5f.close()
#
# h5f = h5py.File('train_data_shortTrial.h5', 'r')
# samples = h5f['full_samples'][:]
# labels = h5f['labels_target'][:]
# print(labels[0])

