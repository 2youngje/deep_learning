from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

class AffineFunction:
    def __init__(self, input_size):
        np.random.seed(8)
        self.w = np.random.randn(2)
        self.b = np.random.randn()
        # self.input = None
        # self.grad_w = None
        # self.grad_b = None

    def __call__(self, x):
        self.x = x
        # z = np.dot(self.w, x) + self.b
        # return z
        self.z = np.dot(self.w, x) + self.b
        return self.z

    def backward(self, grad, learning_rate):
        # self.grad_w = self.x * grad
        # self.grad_b = grad
        # grad_input = self.w * grad
        self.grad_w = self.x
        self.grad_b = 1

        # update
        self.update_params(learning_rate, grad)
        # return grad_input

    def update_params(self, learning_rate, grad):
        self.w -= learning_rate * self.grad_w * grad
        self.b -= learning_rate * self.grad_b * grad

class Sigmoid:
    def __init__(self):
        self.x = None
        self.output = None

    def __call__(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-self.x))
        return self.output

    def backward(self, grad):
        grad_input = np.exp(-self.x) / np.square(1 + np.exp(-self.x)) * grad
        return grad_input

class BCE:
    def __init__(self):
        self.y = None
        self.pred_y = None

    def __call__(self, y, pred_y):
        self.y = y
        self.pred_y = pred_y
        loss = -((self.y * np.log(self.pred_y)) + (1 - self.y) * np.log(1 - self.pred_y))
        return loss

    def backward(self):
        grad_pred_y = (self.pred_y - self.y) / (self.pred_y * (1 - self.pred_y))
        return grad_pred_y

class Function:
    def __init__(self, input_size):
        self.affine = AffineFunction(input_size)
        self.sigmoid = Sigmoid()
        self.bce = BCE()

    def __call__(self, x, target):
        z_affine = self.affine(x)
        y_sigmoid = self.sigmoid(z_affine)
        loss_bce = self.bce(target, y_sigmoid)

        return z_affine, y_sigmoid, loss_bce

    def backward(self, grad_bce, learning_rate):
        grad_sigmoid = self.sigmoid.backward(grad_bce)
        grad_affine = self.affine.backward(grad_sigmoid, learning_rate)

        # self.affine.update_params(learning_rate)

        return grad_affine
#
# # 데이터 생성
# X_train, _ = make_blobs(n_samples=100, centers=2, random_state=42)
# y_train = np.random.randint(2, size=100)
#
# # 학습 파라미터
# input_size = 2
# learning_rate = 0.01
# epochs = 1000
#
# # 모델 초기화
# custom_function = Function(input_size)
#
# # 학습 루프
# for epoch in range(epochs):
#     total_loss = 0
#
#     # 미니배치 학습
#     for x_input, y_label in zip(X_train, y_train):
#         z, y, loss = custom_function(x_input)
#
#         # 역전파 및 파라미터 업데이트
#         grad_bce = custom_function.bce.backward()
#         custom_function.backward(grad_bce, learning_rate)
#
#         total_loss += loss
#
#     # 에포크마다 손실 출력
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {total_loss}")
#
# # 학습된 모델을 사용하여 예측
# X_test, _ = make_blobs(n_samples=2, centers=2, random_state=42)
# for x_test in X_test:
#     _, prediction, _ = custom_function(x_test)
#     print(f"Input: {x_test}, Prediction: {prediction}")