import numpy as np
import matplotlib.pyplot as plt
import cv2

gray_img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)

white = 255
black = 0

white_patch = np.array(white).reshape(1, -1).repeat(10, axis=1)
black_patch = np.array(black).reshape(1, -1).repeat(10, axis=1)

top_img = np.hstack([white_patch, black_patch]).repeat(10, axis=0)
bottom_img = np.hstack([black_patch, white_patch]).repeat(10, axis=0)

data = np.tile(np.vstack([top_img, bottom_img]), reps=[2, 2])

filter_x = np.array([(-1, 0, +1), (-2, 0, +2), (-1, 0, +1)])
filter_y = np.array([(+1, +2, +1), (0, 0, 0), (-1, -2, -1)])

W = 3
H_, W_ = data.shape
L_H = H_ - W + 1
L_W = W_ - W + 1

result_x = np.zeros(np.prod([L_H, L_W])).reshape((L_H, L_W))
result_y = np.zeros(np.prod([L_H, L_W])).reshape((L_H, L_W))

for idx_h in range(L_H):
    for idx_w in range(L_W):
        window = data[idx_h:idx_h + W,idx_w:idx_w + W] # (3, 3)
        co_x_value = (window * filter_x).sum()
        result_x[idx_h][idx_w] = co_x_value
        co_y_value = (window * filter_y).sum()
        result_y[idx_h][idx_w] = co_y_value

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(result_x, cmap="gray")
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.show()

cv2.imshow('gray_img', gray_img)
cv2.waitKey()
cv2.destroyAllWindows()