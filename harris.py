import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc

def harrisCorners(gray):
    
    # harris corner - 60px is a good radius for 1920x1080 video
    corn = cv2.cornerHarris(gray, 3, 29, 0.04)
    
    corners = []
    globalmax = corn.max()

    # filters out matches that are too close to each other
    for y in range(0, corn.shape[0]):
        for x in range(0, corn.shape[1]):
            val = corn[y,x]
            if val > 0.01 * globalmax:

                b = max(y - 10, 0)
                t = min(y + 11, corn.shape[0])
                l = max(x - 10, 0)
                r = min(x + 11, corn.shape[1])
                
                localmax = np.amax(corn[b:t, l:r])
                
                if val == localmax:
                    corners.append(np.array([y,x]))
                    
    return np.asarray(corners)

def displayHarris(img, corners):
    
    newImg = np.copy(img)
    
    for i in range (0, corners.shape[0]):
        cv2.circle(newImg,
                  (corners[i,1], corners[i,0]),
                  10,
                  (255, 0, 0),
                  -1)
    
    plt.imshow(newImg)
    plt.show()
    