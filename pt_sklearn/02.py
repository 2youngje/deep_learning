from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

n_samples = 300
X,y = make_moons(n_samples=n_samples, noise=0.2)

fig,ax = plt.subplots(figsize=(10,10))

plt.xlabel('X')
plt.ylabel('y')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()