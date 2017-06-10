
import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc

def computeTransform(octave, inverse, shift):
    
    if(shift == 0):
        groundTruth = np.float32(np.array([[1000,200],
                           [1060, 560],
                           [1510, 560],
                           [1450, 200]]))
    elif(shift == 1):
        groundTruth = np.float32(np.array([[1030,200],
                           [1030, 560],
                           [1480, 560],
                           [1480, 200]]))
    elif(shift == 2):
        groundTruth = np.float32(np.array([[1060,200],
                           [1000, 560],
                           [1450, 560],
                           [1510, 200]]))
    elif(shift == 3):
        groundTruth = np.float32(np.array([[1000,200],
                           [1060, 560],
                           [1450, 560],
                           [1510, 200]]))
    else:
        groundTruth = np.float32(np.array([[1060,200],
                           [1000, 560],
                           [1510, 560],
                           [1450, 200]]))
    
    l1 = octave[0]
    l2 = octave[4]
    
    if(inverse == 0):
        warped = np.float32(np.array([[l1[1][1], l1[1][0]],
                      [l1[0][1], l1[0][0]],
                      [l2[0][1], l2[0][0]],
                      [l2[1][1], l2[1][0]]]))
    else:
        warped = np.float32(np.array([[l1[0][1], l1[0][0]],
                      [l1[1][1], l1[1][0]],
                      [l2[1][1], l2[1][0]],
                      [l2[0][1], l2[0][0]]]))
    
    transform = cv2.getPerspectiveTransform(warped, groundTruth)
    return transform
    
def applyTransform(transform, img):
    warped = cv2.warpPerspective(img, transform, (3000, 1200))
    return warped

def displayTransform(img):
    plt.imshow(img)
    plt.show()