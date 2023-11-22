import numpy as np
import matplotlib.pyplot as plt

img1 = np.arange(0,255,1).reshape(1,-1) # 255, -> (1,255) -> (100, 255)
img1 = img1.repeat(100, axis=0)
img2 = np.arange(255,0,-1).reshape(1,-1)
img2 = img2.repeat(100, axis=0)
img = np.vstack([img1,img2])

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img,cmap='gray')

ax.tick_params(left=False, labelleft=False,
               bottom=False, labelbottom=False)
plt.show()