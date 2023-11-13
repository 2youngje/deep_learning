import numpy as np
import matplotlib.pyplot as plt
class SigmoidFuntion:
    def __call__(self, z):
        a = 1 / (1+ np.exp(-z))
        return a

sig = SigmoidFuntion()

print(sig.forward(-5))
print(sig.forward(-3))
print(sig.forward(0))
print(sig.forward(-3))
print(sig.forward(-5))

# make data
x = np.arange(-5, 5, 0.1)
y = sig.forward(x)


fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
ax.yaxis.grid(True)

plt.axvline(0.0, color='yellow')
plt.xlabel('x')
plt.ylabel('y')
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

plt.show()