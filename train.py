from torch.utils.data import random_split, DataLoader
from model import *
from Dataloader import Dataset

# prepare dataset
data_set = Dataset("./data.pkl")
train_len = int(len(data_set) * 0.7)
train_set, test_set = random_split(data_set, [train_len, len(data_set) - train_len])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# define training options
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChartNet(30, 30, 512, 512).to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

# start training
for epoch in range(10):
    for index in enumerate(train_loader):
        inputs = data_set.data[index]["input"]
        labels = data_set.data[index]["label"]
        optim.zero_grad()
        outputs = model(inputs)
        running_loss = loss(outputs, labels)
        running_loss.backward()
        optim.step()
        print(f"epoch:{epoch}, index:{index}, loss: {running_loss}")

# save model
torch.save(model.state_dict(), "checkpoints/a.pth")
