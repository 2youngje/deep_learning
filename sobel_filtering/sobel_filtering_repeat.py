import numpy as np

data = np.arange(6).reshape(2,3)
print(data)

print("repeat(axis=0):\n",
      np.repeat(data,repeats=3,axis=0))
print(data.repeat(repeats=3,axis=0))

print("repeat(axis=1):\n",
      np.repeat(data,repeats=3,axis=1))
print(data.repeat(repeats=3,axis=1))
print("repeat(axis=0) and (axis=1):\n",
      np.repeat(np.repeat(data,repeats=2,axis=0),
                repeats=3, axis=1))
print(data.repeat(repeats=2,axis=0).repeat(repeats=3,axis=1))
