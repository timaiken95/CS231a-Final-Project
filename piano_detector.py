import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc
import os

from piano_utils import *
from harris import *
from straightlines import *
from octaves import *
from perpendiculars import *
from octave_transform import *
from warp import *

def detectPiano(originalImage, display=False):
    
    originalCopy = np.copy(originalImage)
    
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    grayImage = np.float32(grayImage)
    
    grayCopy = np.copy(grayImage)
    
    plt.imshow(originalImage)
    plt.show()
    
    currSize = 500.0
    
    for i in range(0, 10):

        resize = min(currSize / originalCopy.shape[1], currSize / originalCopy.shape[0])
        grayImage = misc.imresize(grayCopy, resize)
        originalImage = misc.imresize(originalCopy, resize)
        currSize *= 1.2
        
        if originalImage.shape[0] > 2500 or originalImage.shape[1] > 2500:
            break
    
        corners = harrisCorners(grayImage)
        if display: 
            displayHarris(originalImage, corners)

        lines = fitStraightLines(corners, originalImage)
        if display: 
            displayStraightLines(lines, originalImage)
        
        if len(lines) < 2:
            continue
            
        perpendiculars = fitPerpendiculars(lines, grayImage)
        
        if display: 
            displayPerpendiculars(perpendiculars, originalImage)

        octaves = detectKeyPattern(perpendiculars)
        if display: 
            displayOctaves(originalImage, octaves)

        numOctaves = 0
        for octave in octaves:
            
            minVal = 100
            bestTransform = np.array([])
            
            for flip in range(0, 2):
                for shift in range(0,5):
                    transform = computeTransform(octave, flip, shift)
                    warpedim = applyTransform(transform, originalImage)
                    if display: 
                        displayTransform(warpedim)
                    norm = testWarped(warpedim)
                    if display: 
                        displayWarpTest(warpedim)
                        print(norm)
                        
                    if norm < 50 and norm < minVal:
                        
                        minVal = norm
                        bestTransform = transform
                        if display:
                            displayOctavesOnWarped(warpedim)
            
            if minVal < 50:
                displayOctavesOnOriginal(originalImage, bestTransform)   
                numOctaves += 1   


        print("Number of octaves detected: " + str(numOctaves))