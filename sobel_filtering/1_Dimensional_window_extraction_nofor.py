import numpy as np

np.random.seed(0)

data = np.random.randint(-1,2,(10,))
filter_ = np.array([-1,1,-1])
L, F = len(data), len(filter_)

L_ = L - F + 1

filter_idx = np.arange(F).reshape(1,-1)
window_idx = np.arange(L_).reshape(1,-1)

idx_arr = filter_idx + window_idx

window_mat = data(idx_arr)

print(window_mat.shape, filter_.shape)

correlations = np.matmul(window_mat,filter_.reshape(-1,1))

print(correlations)