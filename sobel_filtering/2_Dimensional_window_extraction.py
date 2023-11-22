import numpy as np

data1 = 10*np.arange(1, 8).reshape(1,-1)
data2 = 10*np.arange(5).reshape(-1,1)

data = data1+data2

print(data,'\n')

H, W = data.shape
F = 3
H_ = H - F + 1
W_ = W - F + 1

for h_idx in range(H_):
    for w_idx in range(W_):
        print(data[h_idx : h_idx + F,
              w_idx : w_idx + F])

    print()

# WIDTH data 길이: 7

# WIDTH 윈도우 길이: 3
#
# WIDTH_윈도우_개수: 5
#
# HEIGHT data개수: 5

# HEIGHT 윈도우 개수: 3
#
# HEIGHT_윈도우_개수: 3