'''
this file is used to implement the model.
'''
import torch
import torch.nn as nn


class ChartNet(nn.Module):
    def __init__(self, fc_feature, audio_feature, hidden_dim, output_dim):
        super(ChartNet, self).__init__()

        # Define the dimensions of the layers
        self.audio_feature = audio_feature
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the CNN layer
        # input: [beat_num, 1, 128, 87], output: [beat_num, audio_feature]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=600, out_features=fc_feature)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=fc_feature, out_features=audio_feature)

        # Define the BiLSTM layer
        # input: [beat_num, audio_feature], output: [1, output_dim]
        self.bilstm = nn.LSTM(audio_feature, hidden_dim, bidirectional=True)

        # Define the output layer
        # Multiply by 2 to account for both forward and backward LSTM
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # the shape of x should be [beat_num, 1, 128, 87]
        # beat_num is number of the beats, 1 means only one channel in MFCC,
        # 128 means n_mel, 87 means the number of columns in 2 seconds in MFCC
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output, _ = self.bilstm(x)
        return self.fc(output)
