import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc

def detectKeyPattern(perpendiculars):
    
    perpSorted = sorted(perpendiculars, cmp=compareLines)
    
    prelimOctaves = [] 
    for i in range(0, len(perpSorted) - 4):
        
        p1 = np.asarray(perpSorted[i][0])
        p2 = np.asarray(perpSorted[i + 1][0])
        p3 = np.asarray(perpSorted[i + 2][0])
        p4 = np.asarray(perpSorted[i + 3][0])
        p5 = np.asarray(perpSorted[i + 4][0])
        
        dist1 = np.linalg.norm(p2 - p1)
        dist2 = np.linalg.norm(p3 - p2)
        dist3 = np.linalg.norm(p4 - p3)
        dist4 = np.linalg.norm(p5 - p4)
            
        avgSmall = np.mean(np.array([dist1, dist3, dist4]))
        avgLarge = np.mean(np.array([dist2]))
            
        residual = np.abs(dist1 - avgSmall) + np.abs(dist3 - avgSmall) + np.abs(dist4 - avgSmall)
        residual += np.abs(dist2 - avgLarge)
        residual += np.abs(dist2 - avgSmall * 1.5)
        residual += np.abs(dist1 - 0.666 * avgLarge) + np.abs(dist3 - 0.666 * avgLarge) + np.abs(dist4 - 0.666 * avgLarge)

        if residual < 200:
            prelimOctaves.append([residual, perpSorted[i], perpSorted[i + 1], perpSorted[i + 2], perpSorted[i + 3], perpSorted[i + 4]])
                
    prelimOctaves = sorted(prelimOctaves, cmp=compareResiduals)
        
    usedPoints = []
    octaves = []
    for r in prelimOctaves:
            
        toAppend = True
        for i in range(1, len(r)):
            point = r[i]
            if any((x[0][0] == point[0][0] and x[0][1] == point[0][1]) for x in usedPoints):
                toAppend = False
        if toAppend:
            octaves.append(r[1:])
            for i in range (1, len(r)):
                usedPoints.append(r[i])
                    
    return octaves

def compareResiduals(item1, item2):
    if item1[0] < item2[0]:
        return -1
    return 1

def compareLines(l1, l2):
    if l1[0][0] * l1[0][1] < l2[0][0] * l2[0][1]:
        return -1
    return 1
    

def displayOctaves(img, octaves):
    
    newImg = np.copy(img)
    
    start = 0
    end = 250   
    
    for octave in octaves:
        
        line1 = octave[0]
        line5 = octave[4]
        
        pts = np.int32(np.asarray([line1[0], line1[1], line5[1], line5[0]]))
        pts = np.fliplr(pts)
        
        rgbl=[end,0,start]
        
        end -= 50
        start += 50
        
        cv2.fillPoly(newImg, [pts], rgbl)
    
    plt.imshow(img)
    plt.imshow(newImg, alpha=0.5)
    plt.show()
