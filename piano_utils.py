import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import misc

# http://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(int(dYa),int(dXa)),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(int), itbuffer[:,0].astype(int)]

    return itbuffer

def testIntersection(v1start, v1end, v2start, v2end):
    a1 = v1end[1] - v1start[1]
    b1 = v1start[0] - v1end[0]
    c1 = (v1end[0] * v1start[1]) - (v1start[0] * v1end[1])
    d1 = (a1 * v2start[0]) + (b1 * v2start[1]) + c1
    d2 = (a1 * v2end[0]) + (b1 * v2end[1]) + c1
    if (d1 > 0 and d2 > 0):
        return False
    if (d1 < 0 and d2 < 0):
        return False
    
    a2 = v2end[1] - v2start[1];
    b2 = v2start[0] - v2end[0];
    c2 = (v2end[0] * v2start[1]) - (v2start[0] * v2end[1]);
    d1 = (a2 * v1start[0]) + (b2 * v1start[1]) + c2;
    d2 = (a2 * v1end[0]) + (b2 * v1end[1]) + c2;
    
    if (d1 > 0 and d2 > 0):
        return False
    if (d1 < 0 and d2 < 0):
        return False
    if ((a1 * b2) - (a2 * b1) == 0):
        return False
    
    return True

def normalize(arr):
    rng = np.max(arr.flatten()) - np.min(arr.flatten())
    amin = np.min(arr.flatten())
    toReturn = (arr - amin) * 255.0 / rng
    return toReturn

def readVideo(path):
    video = cv2.VideoCapture(path)
    
    while(video.isOpened()):
        ret, frame = video.read()
        
        misc.imsave('images/frame.jpg', frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
            
        break
            
    video.release()
    
    
c = np.array([[960,200],
                [960, 750],
                [1045, 750],
                [1045, 560],
                [1000, 560],
                [1000, 200]])
    
csharp = np.array([[1000,200],
                    [1000, 560],
                    [1060, 560],
                    [1060, 200]])
    
d = np.array([[1060,200],
                [1060, 560],
                [1045, 560],
                [1045, 750],
                [1125, 750],
                [1125, 560],
                [1110, 560],
                [1110, 200]])
                    
dsharp = np.array([[1110,200],
                    [1110, 560],
                    [1170, 560],
                    [1170, 200]])
    
e = np.array([[1170,200],
                [1170, 560],
                [1125, 560],
                [1125, 750],
                [1220, 750],
                [1220, 200]])
    
f = np.array([[1220,200],
                [1220, 750],
                [1305, 750],
                [1305, 560],
                [1260, 560],
                [1260, 200]])
    
fsharp = np.array([[1260,200],
                    [1260, 560],
                    [1320, 560],
                    [1320, 200]])
    
g = np.array([[1320, 200],
                [1320, 560],
                [1305, 560],
                [1305, 750],
                [1390, 750],
                [1390, 560],
                [1360, 560],
                [1360, 200]])
    
gsharp = np.array([[1360,200],
                    [1360, 560],
                    [1420, 560],
                    [1420, 200]])
    
a = np.array([[1420, 200],
                [1420, 560],
                [1390, 560],
                [1390, 750],
                [1465, 750],
                [1465, 560],
                [1450, 560],
                [1450, 200]])
    
bflat = np.array([[1450,200],
                    [1450, 560],
                    [1510, 560],
                    [1510, 200]])
    
b = np.array([[1510, 200],
                [1510, 560],
                [1465, 560],
                [1465, 750],
                [1560, 750],
                [1560, 200]])

normalKeys = [c, d, e, f, g, a, b]
sharpKeys = [csharp, dsharp, fsharp, gsharp, bflat]