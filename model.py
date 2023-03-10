'''
this file is used to implement the model. 
'''
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()

        # Define the dimensions of the layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the BiLSTM layer
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

        # Define the output layer
        # Multiply by 2 to account for both forward and backward LSTM
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Pass the input sequence through the BiLSTM layer
        bilstm_out, _ = self.bilstm(x)

        # Concatenate the hidden states of the forward and backward LSTMs
        # The hidden states are stored in the second dimension of the output tensor (i.e., index 1)
        bilstm_out = torch.cat(
            (bilstm_out[:, :, :self.hidden_dim], bilstm_out[:, :, self.hidden_dim:]), dim=2)

        # Pass the concatenated output through the output layer
        output = self.fc(bilstm_out)

        return output
