import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc

def fitStraightLines(points, img):
    lines = []
    numIters = 10000
    pointsToFit = 4
    inlierThreshold = 10
    minLineThreshold = 12
    
    points_hom = np.ones((points.shape[0], 3))
    points_hom[:,0:2] = points
    
    for i in range(0, numIters):
        
        randomSample = points_hom[np.random.choice(points.shape[0], pointsToFit, replace=False)]
        x = randomSample[:,0]
        y = randomSample[:,1]
        
        line, inliers = getLineAndInliers(x, y, points_hom, inlierThreshold)

        if inliers.shape[0] >= minLineThreshold:
            
            newX = inliers[:,0]
            newY = inliers[:,1]
            
            newLine, newInliers = getLineAndInliers(newX, newY, points_hom, inlierThreshold)
            lines.append(np.vstack([newLine, newInliers]))
    
    res = sorted(lines, cmp=compareInliers)
    
    finallines = [] 
    usedPoints = []
    
    for r in res:
        
        toAppend = True
        for i in range(1, r.shape[0]):
            point = r[i,:]
            if any((x[0] == point[0] and x[1] == point[1] and x[2] == point[2]) for x in usedPoints):
                toAppend = False
                break

        if toAppend:
            finallines.append(r)
            for p in range(1, r.shape[0]):
                usedPoints.append(r[p,:])
    
    return finallines[0:2]
    
def compareInliers(item1, item2):
    if item1.shape[0] > item2.shape[0]:
        return -1
    return 1

def displayStraightLines(lines, img):
    
    copy = np.copy(img)
    
    for line in lines:
        for i in range (1, line.shape[0]):
            cv2.circle(copy,
                  (int(line[i,1]), int(line[i,0])),
                  15,
                  (255, 0, 0),
                  -1)
    
    plt.imshow(copy)
    for line in lines:
        
        x = line[1:,0]
        y = line[1:,1]
        m = line[0,0]
        b = line[0,2]
        
        plt.plot(m*x + b, x, 'b')

    plt.show()
    
def getLineAndInliers(x, y, points, inlierThreshold):
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y)[0]
    
    line = np.array([m, -1, b])
    
    homLine = np.array([m, -1, b]) / np.sqrt(np.square(m) + 1)
    
    rep = np.matlib.repmat(homLine, points.shape[0], 1)
    dot = np.abs(np.sum(np.multiply(points, rep), axis=1))
    inliers = points[dot < inlierThreshold]
    return line, inliers

