import cv2
import numpy as np
from PIL import Image

def buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy):
    map_x = np.zeros((Hd,int(Wd)),np.float32)
    map_y = np.zeros((Hd,int(Wd)),np.float32)
    for y in range(0,int(Hd-1)):
        for x in range(0,int(Wd-1)):
            r = (float(y)/float(Hd))*(R2-R1)+R1
            theta = (float(x)/float(Wd))*2.0*np.pi
            xS = Cx+r*np.sin(theta)
            yS = Cy+r*np.cos(theta)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))
        
    return map_x, map_y

def unwarp(img,xmap,ymap):
    output = cv2.remap(np.asarray(img),xmap,ymap,cv2.INTER_LINEAR)
    result = Image.fromarray(output)
    return result

def omni_re(img):
    Cx = 208
    Cy = 208
    R1x = 250
    R1y = 208 
    R1 = R1x-Cx
    R2x = 420
    R2y = 208
    R2 = R2x-Cx
    Wd = 2.0*((R2+R1)/2)*np.pi
    Hd = (R2-R1)
    Ws = 416
    Hs = 416

    xmap,ymap = buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy)

    result = unwarp(img,xmap,ymap)

    return result
