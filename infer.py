'''
this file is used to infer
'''
from model import *


# Create an instance of the BiLSTM class
model = ChartNet(20, 20, 200, 512)
y = model(torch.randn(3, 1, 128, 87))
print(y.shape)
