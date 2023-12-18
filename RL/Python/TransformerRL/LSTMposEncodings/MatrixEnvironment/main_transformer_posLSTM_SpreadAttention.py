import gymnasium
import numpy as np
import torch
from buffer import *
# from agent_attention_BC import Agent as Agent_Attention
from agent_coordinator_posLSTM_SpreadAttention import Agent as Agent_Coordinator

# from five_hot_encode import *
# from belief_update import *
from MATenv import *

def softmax(vector):
    e = np.exp(vector - np.max(vector))  # Subtract max for numerical stability
    return e / e.sum()

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

if __name__ == '__main__':
    env = MatrixAttentionTask()
    ### Trial Max Length ###
    T_end = 7
    input_dim = T_end*32
    # agent_attention = Agent_Attention(alpha=0.00001, beta=0.00001,
    #             input_dims=input_dim, tau=0.001,
    #             env=env, batch_size=1024,
    #             n_actions=4)
    #
    # agent_TORL = Agent_TORL(alpha=0.00001, beta=0.00001,
    #                                   input_dims=input_dim, tau=0.001,
    #                                   env=env, batch_size=1024,
    #                                   n_actions=3)

    agent_coordinator = Agent_Coordinator(alpha=0.00001, beta=0.00001,
                            input_dims=input_dim, tau=0.001,
                            env=env, batch_size=32,
                            n_actions=3+16)

    n_games = 100000
    best_score = env.reward_range[0]
    score_history = []



    # agent_coordinator.load_models()

    for i in range(n_games):
        state = env.reset()

        ### Used for multiple reasons including reward computation and state construction ###
        checkem = np.zeros((T_end, 5))
        atts = np.zeros((T_end, 4))
        actions = np.zeros((T_end, 7))

        ### initial attention ###
        alpha1 = 0
        alpha2 = 0
        alpha3 = 0
        alpha4 = 0

        ### Construct the observation to the belief model (depends on attention) ###
        obs = np.zeros((T_end, 32))
        # obs[0, 0:4] = alpha1 * state[0:4] + alpha2 * state[4:8] + alpha3 * state[8:12] + alpha4 * state[12:16]
        # obs[0, 4] = alpha1
        # obs[0, 5] = alpha2
        # obs[0, 6] = alpha3
        # obs[0, 7] = alpha4
        obs[env.t, 0:16] = state[0:16]
        obs[env.t, 16:] = 0

        mask = np.ones((T_end,32))
        mask[env.t+1:,:] = 0

        # ptorch_obs_coordinator = torch.tensor(obs, dtype=torch.float32).view(1, -1)


        done = False
        score = 0
        while not done:
            ### coordinator is used to construct the state representation ###
            coordinator_action = agent_coordinator.choose_action(obs.reshape(1, -1), mask)
            lever_action = np.argmax(coordinator_action[0:3])
            sensor_action = sigmoid(coordinator_action[3:]/0.01)
            coordinator_action[3:] = sigmoid(coordinator_action[3:]/0.01)
            # print('coordinator action',coordinator_action.shape)
            # tally[env.t+1,coordinator_action] = tally[env.t+1,coordinator_action] + 1
            # state_reps[env.t,:] = coordinator_action

            ### Construct the state ###
            # current_sensor_planner_state = np.zeros((1,input_dim))
            # current_sensor_planner_state[0,coordinator_action] = 1
            # ptorch_obs_sensor_planner = torch.tensor(coordinator_action, dtype=torch.float32).view(1, input_dim)

            ### get the sensor action taken, sensor state value, and sensor log_prob of the action taken ###
            # sensor_action, sensor_value, sensor_log_prob = local_sensor_agent(ptorch_obs_sensor_planner.view(1, -1))
            # print('ATTENTION',attention_action)

            ### where does the attention go ###
            # if sensor_action == 0:  ### agent focuses on S1
            #     alpha1 = 1.0
            #     alpha2 = 0.0
            #     alpha3 = 0.0
            #     alpha4 = 0.0
            # elif sensor_action == 1:  ### agent focuses on S2
            #     alpha1 = 0.0
            #     alpha2 = 1.0
            #     alpha3 = 0.0
            #     alpha4 = 0.0
            # elif sensor_action == 2:  ### agent focuses on S2
            #     alpha1 = 0.0
            #     alpha2 = 0.0
            #     alpha3 = 1.0
            #     alpha4 = 0.0
            # elif sensor_action == 3:  ### agent focuses on S2
            #     alpha1 = 0.0
            #     alpha2 = 0.0
            #     alpha3 = 0.0
            #     alpha4 = 1.0
            # else:
            #     print('oops')

            ### The Planner Action ###
            # planner_action, planner_value, planner_log_prob = local_planner_agent(ptorch_obs_sensor_planner.view(1, -1))

            next_state, reward_env, done, _ = env.step(lever_action)
            # done_attention = True if env.t >= env.T_end else False
            # done_coordinator = False

            ### Construct the next observation to the coordinator model (depends on attention) ###
            obs_ = np.zeros((T_end, 32))
            obs_ = np.copy(obs)
            obs_[env.t, 0:16] = next_state[:]*sensor_action[:]
            obs_[env.t, 16:] = sensor_action[:]
            obs_[env.t+1:, :] = -100

            mask_ = np.ones((T_end, 32))
            mask_[env.t + 1:, :] = 0


            ### Reward Function ###
            # reward_BC = np.sum(np.abs(b - checkem[env.t - 1]))
            # reward_deltaH =  np.sum(b * np.log(b)) - np.sum(checkem[env.t - 1] * np.log(checkem[env.t - 1]))

            # reward_attention = reward_BC
            # reward_TORL = reward_env*3

            # if np.max(np.abs(action[0:-1])) > 1:
            #     reward += - np.sum(np.abs(action[0:-1])**2)
            # if action[-1] > 1:
            #     reward += - (action[-1]**2 - 1)
            # reward = reward_env*10 - np.sum(np.abs(action))/25
            # reward = reward_env +
            # agent_attention.remember(state_vector, attention_action, reward_attention, state_vector_, done_attention)
            agent_coordinator.remember(obs.reshape(1,-1), mask.reshape(1,-1), coordinator_action, reward_env, obs_.reshape(1,-1), mask_.reshape(1,-1), done)

            agent_coordinator.learn()
            # agent_TORL.learn()
            score += reward_env
            obs = obs_
            mask = mask_
            # state_vector = state_vector_

        if i % 100 == 0:
            agent_coordinator.save_models()

            # for i in range(env.t + 1):
                # print("CHECK EM!", checkem[i], atts[i], actions[i], i, env.cue, env.target, env.num_target_features, reward_env)
        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])


        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)










