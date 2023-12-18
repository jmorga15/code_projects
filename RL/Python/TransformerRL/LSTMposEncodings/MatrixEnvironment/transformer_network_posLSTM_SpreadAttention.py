import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.rnn as rnn_utils

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=4,
            name='critic', chkpt_dir='td3_MAT'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 7
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file =  name+'_td3'

        # Embedding Generator
        self.fc_state_projection = nn.Linear(input_dims, input_dims)
        self.ln_state_projection = nn.LayerNorm(input_dims)

        # GRU Layer
        self.LSTM = nn.LSTM(input_size=self.input_dims, hidden_size=self.input_dims, num_layers=3, batch_first=True)

        # Transformer Single Layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dims*2,
            nhead=8,
            dim_feedforward=2048,  # Can be adjusted
            dropout=0.01,
            batch_first=True  # Ensure this is set
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=3)
        self.ln_transformer = nn.LayerNorm(self.input_dims*2*7)

        self.fc_action_lever_projection = nn.Linear(3, self.input_dims*2)
        self.ln_action_lever_projection = nn.LayerNorm(self.input_dims*2)

        self.fc_action_sensor_projection = nn.Linear(16, self.input_dims * 2)
        self.ln_action_sensor_projection = nn.LayerNorm(self.input_dims * 2)

        self.fc1 = nn.Linear(self.input_dims*2*7 + self.input_dims*4, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer and batch norm
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        # self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - representing Q-values for each action
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.q = nn.Linear(fc2_dims, 1)  # policy action selection


        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action, m):
        state = state.view(-1, self.seq_length, self.input_dims)
        batch_size = state.size(0)
        sliced_state = []

        pos_state, (_, _) = self.LSTM(state)

        state = F.sigmoid(self.ln_state_projection(self.fc_state_projection(state)))
        # print(state.shape)
        # Adding Gaussian noise
        std = 0.001  # Adjust this value as needed
        noise = torch.randn_like(state) * std
        state = state + noise
        pos_state = pos_state + noise

        state = state * m.view(-1, self.seq_length, self.input_dims)
        pos_state = pos_state * m.view(-1, self.seq_length, self.input_dims)
        # Concatenate state and pos_state along the feature dimension
        state_pos = torch.cat((state, pos_state), dim=-1)

        # print(state[0])

        # Create a mask where True indicates a position to be ignored (zeros in this case)
        # Check across the input_dim dimension to create a 2D mask
        mask = (state_pos == 0).all(dim=-1)  # Creates a mask of shape [batch_size, seq_len]

        # Ensure the mask is of the correct shape for the transformer
        # For nn.TransformerEncoder, mask should be [seq_len, batch_size]
        # mask = mask.permute(1, 0)  # Switching dimensions if necessary

        # Passing through Transformer
        # (1, seq_len, gru_hidden_dim)
        # Assuming you have an instance of nn.TransformerEncoder
        # Assuming you have an instance of nn.TransformerEncoder
        transformer_out = self.transformer_encoder(state_pos, src_key_padding_mask=mask)
        # Normalize transformer output
        transformer_out = self.ln_transformer(transformer_out.view(batch_size, -1))

        # transformer_out = self.transformer_encoder(sliced_state)
        # transformer_out = transformer_out.view(1,-1)
        #
        # # Aggregate across the sequence dimension
        # transformer_out = torch.mean(transformer_out, dim=1)

        # Concatenating along the second dimension (dim=1)
        # LSTM_out = LSTM_out.view(-1, self.hidden_dim)
        # print('WTFFFFF',LSTM_out.shape)
        action = action.view(-1,self.n_actions)
        action_lever = F.relu(self.ln_action_lever_projection(self.fc_action_lever_projection(action[:,0:3])))
        action_sensor = F.relu(self.ln_action_sensor_projection(self.fc_action_sensor_projection(action[:, 3:])))
        # print('WTFFFFF', action.shape)
        state_action = torch.cat((transformer_out, action_lever, action_sensor), dim=1)

        x = F.elu(self.ln1(self.fc1(state_action)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        q = self.q(x)


        return q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims=32, hidden_dim = 256, fc1_dims=128, fc2_dims=64, n_actions=4, name='Actor', chkpt_dir='td3_MAT'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 7
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        # Embedding Generator
        self.fc_state_projection = nn.Linear(input_dims, input_dims)
        self.ln_state_projection = nn.LayerNorm(input_dims)

        # GRU Layer
        self.LSTM = nn.LSTM(input_size=self.input_dims, hidden_size=self.input_dims, num_layers=3, batch_first=True)

        # Transformer Single Layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dims*2,
            nhead=8,
            dim_feedforward=2048,  # Can be adjusted
            dropout=0.01,
            batch_first=True  # Ensure this is set
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=3)



        self.fc1 = nn.Linear(self.input_dims*2*7, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer and batch norm
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        # self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - representing Q-values for each action
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.pi_lever = nn.Linear(fc2_dims, 3)  # policy action selection


        self.fc1_alpha = nn.Linear(self.input_dims*2*7, hidden_dim)
        self.ln1_alpha = nn.LayerNorm(hidden_dim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer and batch norm
        self.fc2_alpha = nn.Linear(hidden_dim, fc1_dims)
        self.ln2_alpha = nn.LayerNorm(fc1_dims)
        # self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - representing Q-values for each action
        self.fc3_alpha = nn.Linear(fc1_dims, fc2_dims)
        self.ln3_alpha = nn.LayerNorm(fc2_dims)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.pi_sensor = nn.Linear(fc2_dims, 16)  # policy action selection

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, m):
        state = state.view(-1, self.seq_length, self.input_dims)
        batch_size = state.size(0)
        sliced_state = []

        pos_state, (_,_) = self.LSTM(state)

        state = F.sigmoid(self.ln_state_projection(self.fc_state_projection(state)))
        # print(state.shape)
        # Adding Gaussian noise
        std = 0.001  # Adjust this value as needed
        noise = torch.randn_like(state) * std
        state = state + noise
        pos_state = pos_state + noise

        state = state*m.view(-1,self.seq_length,self.input_dims)
        pos_state = pos_state*m.view(-1,self.seq_length,self.input_dims)
        # Concatenate state and pos_state along the feature dimension
        state_pos = torch.cat((state, pos_state), dim=-1)

        # print(state[0])

        # Create a mask where True indicates a position to be ignored (zeros in this case)
        # Check across the input_dim dimension to create a 2D mask
        mask = (state_pos == 0).all(dim=-1)  # Creates a mask of shape [batch_size, seq_len]
        
        # Ensure the mask is of the correct shape for the transformer
        # For nn.TransformerEncoder, mask should be [seq_len, batch_size]
        # mask = mask.permute(1, 0)  # Switching dimensions if necessary

        # Passing through Transformer
        # (1, seq_len, gru_hidden_dim)
        # Assuming you have an instance of nn.TransformerEncoder
        # Assuming you have an instance of nn.TransformerEncoder
        transformer_out = self.transformer_encoder(state_pos, src_key_padding_mask=mask)
        # print(transformer_out.shape)
        # transformer_out = self.transformer_encoder(sliced_state)
        # transformer_out = transformer_out.view(1,-1)
        #
        # # Aggregate across the sequence dimension
        # transformer_out = torch.mean(transformer_out, dim=1)

        # Concatenating along the second dimension (dim=1)
        # LSTM_out = LSTM_out.view(-1, self.hidden_dim)
        # print('WTFFFFF',LSTM_out.shape)
        x = F.elu(self.ln1(self.fc1(transformer_out.view(batch_size, -1))))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        alpha = F.elu(self.ln1_alpha(self.fc1_alpha(transformer_out.view(batch_size, -1))))
        alpha = F.elu(self.ln2_alpha(self.fc2_alpha(alpha)))
        alpha = F.elu(self.ln3_alpha(self.fc3_alpha(alpha)))

        # attention allocation
        pi_lever = self.pi_lever(x)

        # attention allocation
        pi_sensor = self.pi_sensor(alpha)

        # Concatenate along the last dimension
        pi = torch.cat((pi_lever, pi_sensor), dim=-1)


        return pi

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))