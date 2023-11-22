import numpy as np
import matplotlib.pyplot as plt

white_patch = 255*np.ones(shape=(10,10)) #(10,10)짜리 흰색 패치 만들기 => 255로 패우기
black_patch = 0*np.ones(shape=(10,10)) #(10,10)짜리 흰색 패치 만들기 => 0으로 패우기

img1 = np.hstack((white_patch,black_patch,white_patch)) #[흰,검,흰]의 (10,10) 이미지 만들기
img2 = np.hstack((black_patch,white_patch,black_patch)) #[검,흰,검]의 (10,10) 이미지 만들기
img3 = np.hstack((white_patch,black_patch,white_patch)) #[흰,검,흰]의 (10,10) 이미지 만들기
img = np.vstack([img1,img2,img3])

fig,ax = plt.subplots(figsize =(9,9))
ax.imshow(img,cmap='gray') # 이미지 띄우기, 흑백이미지를 만들 것이므로 colormap은 'gray'로

ax.tick_params(left=False,labelleft=False,
               bottom=False, labelbottom=False)

plt.show()