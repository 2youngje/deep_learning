import torch.cuda
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt

# nn.Sequential
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        # self.activation1 = nn.Tanh()
        # self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        #
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # self.activation2 = nn.Tanh()
        # self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        #
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        # self.activation3 = nn.Tanh()
        #
        # self.fc1 = nn.Linear(in_features=120, out_features=84)
        # self.activation4 = nn.Tanh()
        #
        # self.fc2 = nn.Linear(in_features=84, out_features=10)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()

        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),

            nn.Linear(in_features=84, out_features=10)
        )


    def forward(self,x):

        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)

        # x = self.conv1(x)
        # x = self.activation1(x)
        # x = self.avg_pool1(x)
        #
        # x = self.conv2(x)
        # x = self.activation2(x)
        # x = self.avg_pool2(x)
        #
        # x = self.conv3(x)
        # x = self.activation3(x)
        #
        # x = x.reshape(x.shape[0], -1)
        #
        # x = self.fc1(x)
        # x = self.activation4(x)
        #
        # x = self.fc2(x)

        return x


BATCH_SIZE = 32

dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
n_samples = len(dataset)

LR = 0.003
EPOCH = 10

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

model = Model().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)

losses, accs = [],[]
for epoch in range(EPOCH):
    epoch_loss, n_corrects = 0., 0
    for X_,y_ in tqdm(dataloader):
        X_,y_ = X_.to(DEVICE), y_.to(DEVICE)

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

#input_tensor = torch.randn(size=(10, 1, 28, 28))

# CNN = Model()
# a = CNN.forward(dataset)
#
# print(a.shape)