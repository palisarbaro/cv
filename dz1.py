import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def get_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    x, y, w, h = faces[0]
    x1 = int(x + w * 1.1)
    y1 = int(y + h * 1.1)
    x -= int(w * 0.1)
    y -= int(h * 0.1)
    return img[y:y1,x:x1]

def get_bounds(img):
    bounds = cv2.Canny(img, 50, 80)
    return bounds

def filter_bounds(bounds):
    res = np.zeros(bounds.shape,np.uint8)
    contours, hierarchy = cv2.findContours(bounds.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        max_x = contour[:, 0, 0].max()
        max_y = contour[:, 0, 1].max()
        min_x = contour[:, 0, 0].min()
        min_y = contour[:, 0, 1].min()
        if (max_x - min_x > 10) or (max_y - min_y > 10):
            filtered_contours.append(contour)
    cv2.drawContours(res, filtered_contours, -1, (255))
    return res


img = cv2.imread('pic.png', cv2.IMREAD_COLOR)
img = get_face(img)
cv2.imshow('img',img)

bounds = get_bounds(img)
cv2.imshow('bounds',bounds)

filtered_bounds = filter_bounds(bounds)
cv2.imshow('filtered',filtered_bounds)

kernel=np.array([[1.]*5]*5)
dilated = cv2.dilate(filtered_bounds,kernel)
cv2.imshow('dilated',dilated)

M = cv2.filter2D(dilated, -1, cv2.getGaussianKernel(5,20))/255.
cv2.imshow('gaussed',M)

F1 = cv2.bilateralFilter(img,5,20,20)
cv2.imshow('bilateral',F1)

#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel = np.array([[0,0,0],[0,2,0],[0,0,0]]) - np.ones((3,3))/9
F2 = cv2.filter2D(img, -1, kernel)
cv2.imshow('sharp',F2)


result = np.zeros(img.shape,np.uint8)
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        for c in range(3):
            result[x,y,c] = int(M[x,y]*F2[x,y,c] + (1-M[x,y])*F1[x,y,c])

cv2.imshow('result',result)

cv2.waitKey(0)

