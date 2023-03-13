from torch.utils.data import random_split, DataLoader
from model import *
from Dataloader import Dataset

# prepare dataset
data_set = Dataset("./data.pkl")
train_len = int(len(data_set) * 0.8)
train_set, test_set = random_split(data_set, [train_len, len(data_set) - train_len])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# define training options
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChartNet(600, 500, 512,2, 256).to(device)
loss = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(model.parameters(), lr=0.005)
optim=torch.optim.SGD(model.parameters(), lr=0.01)
model.to(device)
model.train()

# start training
for epoch in range(10):
    running_loss=torch.tensor([0],device=device)
    for index,sample in enumerate(train_loader):
        inputs = sample["input"].to(device)
        labels = sample["label"].squeeze(dim=0).to(device)
        optim.zero_grad()
        outputs = model(inputs)
        temp_loss = loss(outputs, labels)
        temp_loss.backward()
        running_loss=running_loss+temp_loss
        optim.step()
        if index%30==0:
            print(f"epoch:{epoch}, index: {index}, loss: {temp_loss}")
    print(f"epoch: {epoch}, avarage loss: {running_loss/index}")

# save model
torch.save(model.state_dict(), "checkpoints/a.pth")

model.eval()
acc=torch.tensor([0.0],device=device)
for index,sample in enumerate(test_loader):
    inputs = sample["input"].to(device)
    outputs = model(inputs)
    outputs=torch.argmax(outputs,dim=1)
    labels = sample["label"].squeeze(dim=0).to(device)
    labels=torch.argmax(labels,dim=1)
    acc=acc+(outputs==labels).mean(dtype=torch.float)
print(f"acc: {acc/index}")