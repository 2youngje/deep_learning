import torch.cuda
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 32

dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
n_samples = len(dataset)

#class MNIST
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier,self).__init__()

        self.fc1 = nn.Linear(in_features=784,out_features=512)
        self.fc1_act = nn.ReLU()

        self.fc2 = nn.Linear(in_features=512,out_features=128)
        self.fc2_act = nn.ReLU()

        self.fc3 = nn.Linear(in_features=128,out_features=52)
        self.fc3_act = nn.ReLU()

        self.fc4 = nn.Linear(in_features=52,out_features=10)

    def forward(self,x):
        x = self.fc1_act(self.fc1(x))
        x = self.fc2_act(self.fc2(x))
        x = self.fc3_act(self.fc3(x))
        x = self.fc4(x)
        return x

LR = 0.003
EPOCH = 10

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

model = MNIST_Classifier().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr=LR)

losses, accs = [],[]
for epoch in range(EPOCH):
    epoch_loss, n_corrects = 0., 0
    for X_,y_ in tqdm(dataloader):
        X_,y_ = X_.to(DEVICE), y_.to(DEVICE)
        X_ = X_.reshape(BATCH_SIZE, -1)

        pred = model(X_)
        loss = loss_function(pred,y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X_)
        n_corrects += (torch.max(pred, axis=1)[1] == y_).sum().item()

    epoch_loss /= n_samples
    losses.append(epoch_loss)

    epoch_acc = n_corrects / n_samples
    accs.append(epoch_acc)

    print(f"Epoch: {epoch+1}")
    print(f"Loss: {epoch_loss:.4f} - Acc : {epoch_acc:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(losses)
axes[1].plot(accs)

axes[1].set_xlabel("Epoch", fontsize=15)
axes[0].set_ylabel("Cross Entorpy Loss", fontsize=15)
axes[1].set_ylabel("Accuracy", fontsize=15)
axes[0].tick_params(labelsize=10)
axes[1].tick_params(labelsize=10)
fig.tight_layout()
plt.show()