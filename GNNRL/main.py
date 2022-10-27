import gym
import numpy as np
from matplotlib import pyplot as plt
import os


from actor import *
from critic import *
from agent import *
from easy_king_env import *


if __name__ == '__main__':

    agent1k = Agent(alpha=0.000001, beta=0.000001,
                input_dims=8*3+8*8, tau=0.0001,
                env=0, batch_size=128, layer1_size=512, layer2_size=256, layer3_size=64,
                n_actions=40, agent_name = 'agent1k')

    n_games = 2500

    best_score = 0
    score_history = []


    agent1k.load_models()

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
        arm_locations, adjacency = env.reset()

        observation1 = arm_locations[0]
        observation2 = arm_locations[1]
        observation3 = arm_locations[2]
        observation4 = arm_locations[3]

        observation5 = arm_locations[4]
        observation6 = arm_locations[5]
        observation7 = arm_locations[6]
        observation8 = arm_locations[7]



        done = False
        score = 0
        turn = 0
        while not done:


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


            ### Get Raw Outputs (actions of all arms)###
            action1k_raw = agent1k.choose_action(observation1k,adjacency)




            ### construct environment action ###
            action_env = np.zeros((8,5))
            action_env[0,:] = np.round(action1k_raw[0])
            action_env[1,:] = np.round(action1k_raw[1])
            action_env[2,:] = np.round(action1k_raw[2])
            action_env[3,:] = np.round(action1k_raw[3])
            action_env[4,:] = np.round(action1k_raw[4])
            action_env[5,:] = np.round(action1k_raw[5])
            action_env[6,:] = np.round(action1k_raw[6])
            action_env[7,:] = np.round(action1k_raw[7])


            ### get new arm locations ###
            arm_locations, reward, adjacency, done = env.step(action_env)



            observation1_ = arm_locations[0]
            observation2_ = arm_locations[1]
            observation3_ = arm_locations[2]
            observation4_ = arm_locations[3]

            observation5_ = arm_locations[4]
            observation6_ = arm_locations[5]
            observation7_ = arm_locations[6]
            observation8_ = arm_locations[7]



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

            ### action used for learning ###
            action_learn = np.zeros((8,5))
            action_learn[0,:] = action1k_raw[8]
            action_learn[1,:] = action1k_raw[9]
            action_learn[2,:] = action1k_raw[10]
            action_learn[3,:] = action1k_raw[11]
            action_learn[4,:] = action1k_raw[12]
            action_learn[5,:] = action1k_raw[13]
            action_learn[6,:] = action1k_raw[14]
            action_learn[7,:] = action1k_raw[15]

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
            # print('action_env',action_env)
            # print('action1k_raw',action1k_raw)
            if i%10 == 0:
                best_score = avg_score
                agent1k.save_models()
            print('episode ', i, 'score %.2f' % score,
                    'trailing 100 games avg %.3f' % avg_score)
