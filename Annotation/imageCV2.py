import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('lung-tissue.jpg',0)
edges = cv2.Canny(img,100,200)
plt.imshow(edges, cmap='jet')


contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
with open("coords.txt", "w") as f:
    for i in contours:
        f.write(str(i)+"\n")
plt.imsave("test.png", edges)
plt.show()