import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc
from piano_utils import *

def fitPerpendiculars(lines, image):
    line1 = lines[0]
    line2 = lines[1]
    
    perpendiculars = []
    for i in range(1, line1.shape[0]):
        for j in range(1, line2.shape[0]):
            p1 = np.array([line1[i, 1], line1[i, 0]])
            p2 = np.array([line2[j, 1], line2[j, 0]])
            
            pixels = createLineIterator(p1, p2, image)
            if len(pixels) > 0:
                std = np.mean(pixels[:,2]) * np.std(pixels[:,2])
                if(std < 3000):
                    perpendiculars.append([std, line1[i,0:2], line2[j,0:2]])
    
    res = sorted(perpendiculars, cmp=compareStd)
    
    finalPerps = []
    usedPoints = []
    
    for r in res:

        toAppend = True
        if any((x[0] == r[1][0] and x[1] == r[1][1]) for x in usedPoints):
            toAppend = False
        if any((x[0] == r[2][0] and x[1] == r[2][1]) for x in usedPoints):
            toAppend = False
            
        for perp in finalPerps:
            if testIntersection(perp[0], perp[1], r[1], r[2]):
                toAppend = False
                break

        if toAppend:
            finalPerps.append(r[1:])
            usedPoints.append(r[1])
            usedPoints.append(r[2])
    
    return finalPerps

def displayPerpendiculars(perps, image):
    plt.imshow(image, cmap='gray')
    for r in perps:

        pt1 = r[0]
        pt2 = r[1]
        
        plt.plot([pt1[1], pt2[1]], [pt1[0], pt2[0]])
    
    plt.show()
    
    
def compareStd(item1, item2):
    if item1[0] < item2[0]:
        return -1
    return 1
    