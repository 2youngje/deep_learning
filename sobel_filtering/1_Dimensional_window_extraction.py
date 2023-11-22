import numpy as np
import matplotlib.pyplot as plt

data = 10*np.arange(1,11)

window_size = 3
len_data = len(data)

window_count = len_data-window_size+1

for i in range(window_count):
    print(data[i:i+window_size])