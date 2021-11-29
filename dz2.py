import cv2
import numpy as np
import sys
from random import random




def conv(img):
    weights = [[[[random() for k in range(3)] for j in range(3)] for i in range(3)] for m in range(5)]
    b = [random() for i in range(5)]
    shape = img.shape
    out_shape = shape[0] - 2, shape[1] - 2, 5
    res = np.zeros(out_shape, 'float')
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            for M in range(5):
                res[i][j][M] = sum([weights[M][I][J][K] * img[i + I][j + J][K] for K in range(3) for J in range(3) for I in range(3)]) + b[M]
    return res


def norm(img):
    res = np.zeros(img.shape,'float')
    a = [random() for i in range(5)]
    b = [random() for i in range(5)]
    for M in range(5):
        mean = np.mean(img[:,:,M])
        std = np.std(img[:,:,M])
        res[:,:,M] = a[M]*(img[:,:,M]-mean)/std + b[M]
    return res

def max_pooling(img):
    out_shape = img.shape[0]//2, img.shape[1]//2, 5
    res = np.zeros(out_shape,'float')
    for M in range(5):
        for i in range(0,img.shape[0],2):
            for j in range(0, img.shape[1],2):
                res[i//2,j//2,M] = max([img[i,j,M], img[i+1,j,M], img[i,j+1,M], img[i+1,j+1,M]])
    return res

def sm(arr):
    r = np.array([np.e**a for a in arr])
    return r/r.sum()

def softMax(img):
    res = np.zeros(img.shape,'float')
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            res[i,j,:] = sm(img[i,j,:])
    return res


np.set_printoptions(threshold=sys.maxsize)
img = cv2.imread('pic.png', cv2.IMREAD_COLOR)
img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)

print(img.shape)
after_conv = conv(img)
print(after_conv.shape)
after_norm = norm(after_conv)
print(after_norm.shape)
after_relu = np.maximum(0,after_norm)
print(after_relu.shape)
after_pool = max_pooling(after_relu)
print(after_pool.shape)
after_soft = softMax(after_pool)
print(after_soft.shape)


for M in range(5):
    mat = after_soft[:,:,M]
    cv2.imshow(str(M),mat)

cv2.waitKey()