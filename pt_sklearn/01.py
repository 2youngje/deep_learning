from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_samples = 100
X,y = make_blobs(n_samples=n_samples, centers=2, n_features=2,cluster_std=0.5)

fig,ax = plt.subplots(figsize=(10,10))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Generated Data with Two Classes')
plt.xlabel('X')
plt.ylabel('y')
plt.show()