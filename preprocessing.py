'''
this file is used to:
1. compute MFCC and beat guide
2. save the information above into ./data_train
'''
import torch.nn as nn
import torch
import json
import librosa
import numpy as np
hop_length = 512
y, sr = librosa.load("try.ogg")
t, b = librosa.beat.beat_track(y=y, sr=sr, units="samples")
M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
print(M[:, 2000])
print(y.shape, sr)
print(librosa.frames_to_time(M.shape[1]))
print(M.shape)


with open('try.mc', encoding='utf-8') as f:
    data = json.load(f)

print(data.keys())
print(data['time'])

# for item in data['array']:
#     print(item)
rnn = nn.LSTM(10, 20, 2, bidirectional=True)
input = torch.randn(7, 3, 10)
h0 = torch.randn(2*2, 3, 20)
c0 = torch.randn(2*2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape, hn.shape, cn.shape)

# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)
