from torch.utils.data import random_split, DataLoader
from model import *
from Dataloader import Dataset
from tqdm import tqdm

# prepare dataset
data_set = Dataset("./data.pkl")
train_len = int(len(data_set) * 0.8)
train_set, test_set = random_split(
    data_set, [train_len, len(data_set) - train_len])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# compute proper weight in CrossEntropyLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight = torch.tensor([20, 100, 5, 5], device=device).to(torch.float)

# define training options
model = ChartNet(600, 500, 512, 2, 16).to(device)
model.load_state_dict(torch.load('checkpoints/e_8.pth', map_location=device))

loss = nn.CrossEntropyLoss(weight=weight)
# optim = torch.optim.Adam(model.parameters(), lr=0.05)
optim = torch.optim.SGD(model.parameters(), lr=0.1)
model.to(device)
model.train()

# start training
for epoch in range(30):
    running_loss = torch.tensor([0], device=device)
    for index, sample in enumerate(tqdm(train_loader)):
        inputs = sample["input"].to(device)
        labels = sample["label"].squeeze(dim=0).to(device)
        optim.zero_grad()
        outputs = model(inputs)
        outputs = torch.reshape(outputs, [-1, 4, 4])
        temp_loss = loss(outputs, labels.to(torch.long))
        temp_loss.backward()
        running_loss = running_loss + temp_loss
        optim.step()
    print(f"epoch: {epoch}, avarage loss: {running_loss/index}")

# save model
torch.save(model.state_dict(), "checkpoints/e_9.pth")
print(f"finished")
