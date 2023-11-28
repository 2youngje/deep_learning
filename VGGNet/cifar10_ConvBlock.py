import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# ConvBlock 정의

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ConvBlock, self).__init__()

        # 처음 conv layer는 in_channels를 사용
        self.layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]

        # n_layers가 2 이상일 때 동작하는 코드
        for _ in range(n_layers - 1):
            self.layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                         kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            in_channels = out_channels

        # 마지막에 max pooling 추가
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # list에 들어있는 layer를 풀어 nn.Sequential에 입력
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

# 데이터 로드

transform = ToTensor()
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 생성 및 학습

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv_block = ConvBlock(3, 64, 2)  # 입력 채널은 3 (RGB), 2개의 convolutional layer 사용
        self.fc = nn.Linear(64 * 16 * 16, 10)  # CIFAR-10 이미지 크기 (32, 32)에 맞게 수정

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CIFAR10Model()

# 모델 요약 출력
summary(model, input_size=(3, 32, 32))

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 학습 루프
epochs = 5

# 결과 기록을 위한 리스트
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 에폭별 손실 기록
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # 검증 데이터에 대한 성능 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

# 손실과 정확도 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()