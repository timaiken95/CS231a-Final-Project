import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc
from piano_utils import *

def displayOctavesOnWarped(warped):
    
    newImg = np.copy(warped)
    
    sharpKeysCopy = list(sharpKeys)
    normalKeysCopy = list(normalKeys)
    
    for key in sharpKeysCopy:
        pts = np.int32(key)
        
        rgbl=[255,0,0, 125]
        
        cv2.fillPoly(newImg, [pts], rgbl)
        
    start = 1
    end = 250
    for key in normalKeysCopy:
        pts = np.int32(key)
        
        
        rgbl=[0,end,start]
        
        end -= 40
        start += 40
        
        cv2.fillPoly(newImg, [pts], rgbl)
    
    plt.imshow(warped)
    plt.imshow(newImg, alpha=0.5)
    plt.show()
    
    
    
def displayOctavesOnOriginal(img, transform):
    sharpKeysCopy = list(sharpKeys)
    normalKeysCopy = list(normalKeys)
    
    for i, key in enumerate(sharpKeysCopy):
        sharpKeysCopy[i] = cv2.perspectiveTransform(np.float32(key[None, :, :]),
                                                    np.float32(np.linalg.inv(transform)))
    
    for i, key in enumerate(normalKeysCopy):
        normalKeysCopy[i] = cv2.perspectiveTransform(np.float32(key[None, :, :]),
                                                    np.float32(np.linalg.inv(transform)))
    
    newImg = np.copy(img)
    
    for key in sharpKeysCopy:
        pts = np.int32(key)
        
        rgbl=[255,0,0, 125]
        
        cv2.fillPoly(newImg, [pts], rgbl)
        
    start = 1
    end = 250
    for key in normalKeysCopy:
        pts = np.int32(key)
        
        
        rgbl=[0,end,start]
        
        end -= 40
        start += 40
        
        cv2.fillPoly(newImg, [pts], rgbl)
    
    plt.imshow(img)
    plt.imshow(newImg, alpha=0.5)
    plt.show()

def testWarped(warpedImg):
    gray = np.float32(cv2.cvtColor(warpedImg, cv2.COLOR_BGR2GRAY))
    octaveSegment = gray[200:750, 960:1560]
    groundTruth = misc.imread('groundtruthresized.png', 'L')
    
    sub = np.abs(normalize(octaveSegment) - normalize(groundTruth))
    norm = np.mean(sub.flatten())

    return norm

def displayWarpTest(warpedImg):
    gray = np.float32(cv2.cvtColor(warpedImg, cv2.COLOR_BGR2GRAY))
    octaveSegment = gray[200:750, 960:1560]
    groundTruth = misc.imread('groundtruthresized.png', 'L')
    
    sub = np.abs(normalize(octaveSegment) - normalize(groundTruth))
    
    plt.imshow(sub, cmap='gray')
    plt.show()