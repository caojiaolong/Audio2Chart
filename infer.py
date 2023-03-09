'''
this file is used to infer
'''
from model import *

# Define the input sequence dimensions, hidden layer dimensions, and output dimensions
input_dim = 10
hidden_dim = 20
output_dim = 5

# Create an instance of the BiLSTM class
model = BiLSTM(input_dim, hidden_dim, output_dim)

# Generate some sample input data
seq_len = 5
batch_size = 3
input_data = torch.randn(seq_len, batch_size, input_dim)

# Pass the input data through the model
output = model(input_data)

# Print the output tensor shape
print(f"{output}\n")
print(output.shape)
