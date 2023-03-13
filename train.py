from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChartNet(30, 30, 512, 512).to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epoch in range(10):
    running_loss = 0.0
