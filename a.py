import cv2 
import numpy as np 
from numba import jit
from logging import getLogger

def makeMG(_img, _N):
    res = np.zeros((_img.shape[0], _img.shape[1]))
    res.fill(255)

    q_bit = _N // 2
    halfpitch = _N // 2
    br_div = 256 // q_bit
    
    for j in range(0, res.shape[0], _N):
        for i in range(0, res.shape[1], _N):
            br = _img[j][i]
            
            pos = br // br_div
            for l in range(_N):
                for k in range(halfpitch):
                    if j + l >= _img.shape[0] or i + pos + k >= _img.shape[1]:
                        break
                    res[j + l][i + pos + k] = 0
            
    return res    

def makeRefMG(_img,_N):
    res = np.zeros((_img.shape[0], _img.shape[1], 4))
    res.fill(255)

    halfpitch = _N // 2

    for j in range(res.shape[0]):
        white = False
        for i in range(0, res.shape[1], halfpitch):
            if white:
                for k in range(halfpitch):
                    if i + k >= res.shape[1]:
                        break
                    res[j][i + k] = [255, 255, 255, 0]
                    white = False
            else:
                for k in range(halfpitch):
                    if i + k >= res.shape[1]:
                        break
                    res[j][i + k] = [0, 0, 0, 255]
                    white = True
    return res

def main():
    fn = "origin2.jpg"
    img = cv2.imread(fn, 0)
    N = 8
    
    moire_img = makeMG(img, N)
    ref_img = makeRefMG(img, N)
    
    cv2.imwrite("./test2.png", moire_img)
    cv2.imwrite("./ref2.png", ref_img)
    
if __name__=="__main__":
    main()