import numpy as np
import matplotlib.pyplot as plt

white_patch = 255*np.ones(shape=(10,10)) #(10,10)짜리 흰색 패치 만들기 => 255로 채우기
gray_patch = 127*np.ones(shape=(10,10)) #(10,10)짜리 회색 패치 만들기 => 127로 채우기
black_patch = 0*np.ones(shape=(10,10)) #(10,10)짜리 흰색 패치 만들기 => 0으로 채우기

img1 = np.hstack((white_patch,gray_patch)) #[흰,검]의 (10,10) 이미지 만들기
img2 = np.hstack((black_patch,white_patch)) #[검,흰]의 (10,10) 이미지 만들기
img3 = np.hstack((gray_patch,black_patch)) #[회,검]의 (10,10) 이미지 만들

img = np.vstack([img1,img3])
img = np.tile(img,reps=[4,4])

fig,ax = plt.subplots(figsize =(8,8))
ax.imshow(img,cmap='gray')

ax.tick_params(left=False,labelleft=False,
               bottom=False, labelbottom=False)

#sobel

plt.show()