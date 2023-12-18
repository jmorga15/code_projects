import gymnasium
import numpy as np
import torch
from buffer import *
# from agent_attention_BC import Agent as Agent_Attention
from agent_coordinator_posLSTM_SpreadAttention import Agent as Agent_Coordinator
# from trainer_network import *
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch as T


# from five_hot_encode import *
# from belief_update import *
from MATenv import *


def softmax(vector):
    e = np.exp(vector - np.max(vector))  # Subtract max for numerical stability
    return e / e.sum()

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output, attn

def compute_qkv(input, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, n_heads):
    d_k = q_weight.size(0) // n_heads
    print('input size: I wish chatgpt would give me code that works',input.size())
    # seq_len, feat_dim = input.size()
    seq_len = 7
    feat_dim = 32

    # Linear transformations and reshaping for multi-head
    query = F.linear(input, q_weight, q_bias).view(1, seq_len, n_heads, d_k).transpose(1, 2)
    key = F.linear(input, k_weight, k_bias).view(1, seq_len, n_heads, d_k).transpose(1, 2)
    value = F.linear(input, v_weight, v_bias).view(1, seq_len, n_heads, d_k).transpose(1, 2)

    return query, key, value

def plot_attention_heatmaps(attn_matrix, sample_index=0):
    # attn_matrix shape: [batch_size, n_heads, seq_length, seq_length]
    n_heads = attn_matrix.shape[1]

    fig, axs = plt.subplots(1, n_heads, figsize=(n_heads * 4, 4))
    fig.suptitle('Attention Heatmaps for Each Head')

    for i in range(n_heads):
        # Select the attention matrix for the specific head and sample
        attn_map = attn_matrix[sample_index, i].detach().cpu().numpy()

        ax = axs[i]
        cax = ax.matshow(attn_map, cmap='viridis')
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        fig.colorbar(cax, ax=ax)

    plt.show()

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))



if __name__ == '__main__':
    env = MatrixAttentionTask()
    ### Trial Max Length ###
    T_end = 7
    input_dim = T_end*32

    # model = TransformerNetwork(0.00001, input_dims=16, output=3)
    # model.load_state_dict(torch.load('BeliefMasterRAU_MultiON_switchingNoise2.pth'))

    agent_coordinator = Agent_Coordinator(alpha=0.00001, beta=0.00001,
                                          input_dims=input_dim, tau=0.001,
                                          env=env, batch_size=32,
                                          n_actions=3+16)
    agent_coordinator.load_models()

    n_games = 1
    best_score = env.reward_range[0]
    score_history = []



    #agent.load_models()

    for i in range(n_games):
        state = env.reset()

        ### Construct the observation to the belief model (depends on attention) ###
        obs = np.zeros((T_end, 32))
        # obs[0, 0:4] = alpha1 * state[0:4] + alpha2 * state[4:8] + alpha3 * state[8:12] + alpha4 * state[12:16]
        # obs[0, 4] = alpha1
        # obs[0, 5] = alpha2
        # obs[0, 6] = alpha3
        # obs[0, 7] = alpha4
        obs[env.t, 0:16] = state[0:16]
        obs[env.t, 16:] = 0.00001

        mask = np.ones((T_end, 32))
        mask[env.t + 1:, :] = 0

        done = False
        score = 0
        while not done:
            ### coordinator is used to construct the state representation ###
            coordinator_action = agent_coordinator.choose_action(obs.reshape(1, -1), mask)
            lever_action = np.argmax(coordinator_action[0:3])
            sensor_action = sigmoid(coordinator_action[3:]/0.01)
            print('sensor_action action, target',sensor_action.reshape(4,4),env.target)

            state = T.tensor(obs, dtype=T.float).to(agent_coordinator.actor.device)

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
            obs_[env.t, 0:16] = next_state[:] * sensor_action[:]
            obs_[env.t, 16:] = sensor_action[:]
            obs_[env.t + 1:, :] = -100

            mask_ = np.ones((T_end, 32))
            mask_[env.t + 1:, :] = 0

            ### Reward Function ###
            # reward_BC = np.sum(np.abs(b - checkem[env.t - 1]))
            # reward_deltaH =  np.sum(b * np.log(b)) - np.sum(checkem[env.t - 1] * np.log(checkem[env.t - 1]))

            # reward_attention = reward_BC
            # reward_TORL = reward_env*3

            # agent_TORL.learn()
            score += reward_env
            obs = obs_
            mask = mask_
            # state_vector = state_vector_

        # if i % 100 == 0:

            # for i in range(env.t + 1):
            #     print("CHECK EM!", checkem[i], atts[i], actions[i], i, env.cue, env.target, env.num_target_features, reward_env)
        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])

        # state = T.tensor(obs, dtype=T.float).to(model.device)
        # Assuming transformer_encoder_layer is your first layer of Transformer
        # weights = model.transformer_encoder_layer.self_attn.in_proj_weight
        # biases = model.transformer_encoder_layer.self_attn.in_proj_bias

        # Split the weights and biases for queries, keys, and values
        # d_model = weights.size(1) // 3
        # q_weight, k_weight, v_weight = weights.chunk(3, dim=0)
        # q_bias, k_bias, v_bias = biases.chunk(3, dim=0)
        #
        # # Assuming state is of shape [1, seq_len, model_d]
        # n_heads = model.transformer_encoder_layer.self_attn.num_heads
        # query, key, value = compute_qkv(state, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, n_heads)
        #
        # # Compute attention
        # _, attn_matrix = scaled_dot_product_attention(query, key, value)
        #
        # # attn_matrix should be of shape [n_heads, 1, seq_len, seq_len]
        # print(attn_matrix.shape)
        # n_heads = model.transformer_encoder_layer.self_attn.num_heads
        #
        # # plot_attention_heatmaps(attn_matrix.transpose(0, 1), sample_index=0)
        #
        # # n_heads = model.transformer_encoder_layer.self_attn.num_heads
        # # print(n_heads)
        # # print(attn_matrix.shape)
        # attn_matrix = attn_matrix.view(1, n_heads, T_end, T_end)
        #
        # # Example usage
        # plot_attention_heatmaps(attn_matrix, sample_index=0)  # for the first sample in the batch









        # state = T.tensor(obs, dtype=T.float).to(agent_coordinator.actor.device)

        # Access the second layer of the Transformer encoder
        first_agent_layer = agent_coordinator.actor.transformer_encoder.layers[0]

        state = state.view(-1, 7, 32)
        batch_size = state.size(0)
        sliced_state = []

        pos_state, (_, _) = agent_coordinator.actor.LSTM(state)

        state = F.sigmoid(agent_coordinator.actor.ln_state_projection(agent_coordinator.actor.fc_state_projection(state)))

        state = state
        pos_state = pos_state
        # Concatenate state and pos_state along the feature dimension
        state_pos = torch.cat((state, pos_state), dim=-1)

        # print(state[0])
        #
        # Create a mask where True indicates a position to be ignored (zeros in this case)
        # Check across the input_dim dimension to create a 2D mask
        mask = (state_pos == 0).all(dim=-1)  # Creates a mask of shape [batch_size, seq_len]

        # Get the weights and biases for the second layer's self-attention mechanism
        weights = first_agent_layer.self_attn.in_proj_weight
        biases = first_agent_layer.self_attn.in_proj_bias

        # Split the weights and biases for queries, keys, and values
        d_model = weights.size(1) // 3
        q_weight, k_weight, v_weight = weights.chunk(3, dim=0)
        q_bias, k_bias, v_bias = biases.chunk(3, dim=0)

        # Assuming state is of shape [1, seq_len, model_d]
        n_heads = first_agent_layer.self_attn.num_heads
        query, key, value = compute_qkv(state_pos, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, n_heads)

        # Compute attention
        _, attn_matrix = scaled_dot_product_attention(query, key, value)

        # attn_matrix should be of shape [n_heads, 1, seq_len, seq_len]
        print(attn_matrix.shape)

        # Reshape the attention matrix for plotting
        attn_matrix = attn_matrix.view(1, n_heads, attn_matrix.size(-1), attn_matrix.size(-1))

        # Example usage
        plot_attention_heatmaps(attn_matrix, sample_index=0)  # for the first sample in the batch















        #
        #
        #
        #
        #
        # # Access the second layer of the Transformer encoder
        # second_layer = model.transformer_encoder.layers[1]
        #
        # # Get the weights and biases for the second layer's self-attention mechanism
        # weights = second_layer.self_attn.in_proj_weight
        # biases = second_layer.self_attn.in_proj_bias
        #
        # # Split the weights and biases for queries, keys, and values
        # d_model = weights.size(1) // 3
        # q_weight, k_weight, v_weight = weights.chunk(3, dim=0)
        # q_bias, k_bias, v_bias = biases.chunk(3, dim=0)
        #
        # # Assuming state is of shape [1, seq_len, model_d]
        # n_heads = second_layer.self_attn.num_heads
        # query, key, value = compute_qkv(state, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, n_heads)
        #
        # # Compute attention
        # _, attn_matrix = scaled_dot_product_attention(query, key, value)
        #
        # # attn_matrix should be of shape [n_heads, 1, seq_len, seq_len]
        # print(attn_matrix.shape)
        #
        # # Reshape the attention matrix for plotting
        # attn_matrix = attn_matrix.view(1, n_heads, attn_matrix.size(-1), attn_matrix.size(-1))
        #
        # # Example usage
        # plot_attention_heatmaps(attn_matrix, sample_index=0)  # for the first sample in the batch

        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)










