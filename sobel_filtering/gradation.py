import numpy as np
import matplotlib.pyplot as plt

img = np.arange(255,0,-255/6).reshape(-1,1)
img = img.repeat(100, axis=0).repeat(300,axis=1)

fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(img,cmap='gray', vmax=255, vmin=0)

ax.tick_params(left=False, labelleft=False,
               bottom=False, labelbottom=False)
plt.show()