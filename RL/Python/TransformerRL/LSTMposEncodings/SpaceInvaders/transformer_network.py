import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim = 256, fc1_dims=64, fc2_dims=32, n_actions=4,
            name='critic', chkpt_dir='td3_MAT'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 5
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file =  name+'_td3'

        self.conv1 = nn.Conv2d(1, 16, kernel_size=16, stride=6, padding=1)  # Example
        self.conv2 = nn.Conv2d(16, 32, kernel_size=8, stride=3, padding=1)  # Example
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)  # Example

        # GRU Layer
        self.LSTM = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True)

        # # Transformer Single Layer
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.input_dims,
        #     nhead=4,
        #     dim_feedforward=2048,  # Can be adjusted
        #     dropout=0.01,
        #     batch_first=True  # Ensure this is set
        # )
        # self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=3)

        self.fc_action_projection = nn.Linear(n_actions, hidden_dim)
        self.ln_action_projection = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(256*3+hidden_dim, hidden_dim)
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

    def forward(self, state, action):
        batch_size = state.size(0)

        ### shape (5,84,84) ###
        state = state.view(-1, 1, 84, 84)

        # Apply convolutional layers
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        # Reshape back to include the sequence length
        state = state.view(batch_size, self.seq_length, -1)
        # print('state shape',state.shape)



        # Passing the sequence through the GRU
        LSTM_out, (h, c) = self.LSTM(state)

        # Concatenating along the second dimension (dim=1)
        # LSTM_out = LSTM_out.view(-1, self.hidden_dim)
        # print('WTFFFFF',LSTM_out.shape)
        # print('action', action.shape)
        action = action.view(-1,self.n_actions)
        action = F.sigmoid(self.ln_action_projection(self.fc_action_projection(action)))
        # print('WTFFFFF', action.shape)
        state_action = torch.cat((h.view(batch_size,256*3), action), dim=1)

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
    def __init__(self, alpha, input_dims=32, hidden_dim = 256, fc1_dims=64, fc2_dims=32, n_actions=4, name='Actor', chkpt_dir='td3_MAT'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 5
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        # Define your convolutional layers here
        # Define your convolutional layers here
        self.conv1 = nn.Conv2d(1, 16, kernel_size=16, stride=6, padding=1)  # Example
        self.conv2 = nn.Conv2d(16, 32, kernel_size=8, stride=3, padding=1)  # Example
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)  # Example

        # GRU Layer
        self.LSTM = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=True)

        # # Transformer Single Layer
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.input_dims,
        #     nhead=4,
        #     dim_feedforward=2048,  # Can be adjusted
        #     dropout=0.01,
        #     batch_first=True  # Ensure this is set
        # )
        # self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=3)

        self.fc1 = nn.Linear(256*3, hidden_dim)
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
        self.pi = nn.Linear(fc2_dims, n_actions)  # policy action selection

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        batch_size = state.size(0)

        ### shape (5,84,84) ###
        state = state.view(-1, 1, 84, 84)

        # Apply convolutional layers
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        # print('state shape', state.shape)

        # Reshape back to include the sequence length
        state = state.view(batch_size, self.seq_length, -1)
        # print('state shape', state.shape)

        # Passing the sequence through the GRU
        LSTM_out, (h,c) = self.LSTM(state)
        # Create a batch index
        batch_size = LSTM_out.shape[0]
        batch_indices = torch.arange(batch_size)

        # LSTM_out = LSTM_out.view(batch_size, -1)
        x = F.elu(self.ln1(self.fc1(h.view(-1,3*256))))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        # attention allocation
        pi = self.pi(x)


        return pi

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))