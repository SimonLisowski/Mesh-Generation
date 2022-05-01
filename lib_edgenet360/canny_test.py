import cv2
import numpy as np
from matplotlib import pyplot as plt

img_files = ['/d02/data/csscnet/SUNCG/Train/0a86348c95f4548e15446232bd187460/00000010_color.jpg',
             '/d02/data/csscnet/NYU/NYUtrain/NYU0005_0000_color.jpg']


for i, img_file in enumerate(img_files):

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img,100,200)
    plt.subplot(221 + i*2),plt.imshow(img) #,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222 + i*2),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()

np.unique(edges)