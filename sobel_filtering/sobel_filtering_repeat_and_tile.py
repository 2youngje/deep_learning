import numpy as np

data = np.arange(5)
print(data)

#np.repeat => 원소별 반복
print("repeat", np.repeat(data, repeats=3))

#np.tile => 전체 패턴 반복
print("tile",np.tile(data, reps=3))