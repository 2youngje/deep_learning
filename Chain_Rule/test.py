import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from model_test import Function
import numpy as np

N_SAMPLES = 100
LR = 0.001
EPOCHS = 30
input_size = 2
X, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=2,
cluster_std=0.5, random_state=0)

''' Instantiation '''
model = Function(input_size)
# compare = lambda y, y_pred: int(y == np.round(y_pred))

epoch_accuracy = []
epoch_loss = []
for epoch in range(EPOCHS):
    total_loss = 0
    total_accuracy = 0
    for X_, y_ in zip(X, y):

        ''' Training '''
        z, y_pred, loss = model(X_, y_)

        grad_bce = model.bce.backward()
        model.backward(grad_bce, LR)
        tmp = np.round(y_pred)
        total_accuracy += int(y_ == int(np.round(y_pred)))
        total_loss += loss

    ''' Metric(loss, accuracy) Calculations '''
    # for문 끝나면 평균 계산
    # 1. loss 100개의 평균 -> 이번 에폭의 평균 loss
    # -> 30개를 모아야됨 (total_loss에)
    epoch_loss.append(total_loss/N_SAMPLES)
    # 2. accuracy 100개의 평균 -> 이번 에폭의 평균 accuracy
    # -> 30개를 모아야됨 (total_accuracy에)
    epoch_accuracy.append(total_accuracy / N_SAMPLES)


''' Result Visualization '''

fig,axes = plt.subplots(2, 1, figsize=(10,10))
axes[0].plot(epoch_loss)
axes[1].plot(epoch_accuracy)
plt.show()