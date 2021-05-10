#!/usr/bin/env python
from __future__ import print_function
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import argparse
import time
import requests
start_time = time.time()
img_logo = cv.imread('logo.jpg', cv.IMREAD_GRAYSCALE)
img_scene = cv.imread('meuble.jpg', cv.IMREAD_GRAYSCALE)
Flogo = img_logo
Fscene = img_scene
width = 1000
height = int(img_logo.shape[0] * width/img_logo.shape[1])
dim = (width, height)
# resize image
img_logo = cv.resize(img_logo, dim, interpolation=cv.INTER_AREA)
height = int(img_scene.shape[0] * width/img_scene.shape[1])
dim = (width, height)
img_scene = cv.resize(img_scene, dim, interpolation=cv.INTER_AREA)
print("load--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
if img_logo is None or img_scene is None:
    print('Could not open or find the images!')
    exit(0)
# -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints_obj, descriptors_obj = detector.detectAndCompute(img_logo, None)
keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
print("pas2--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
# -- Filter matches using the Lowe's ratio test
ratio_thresh = 0.75
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
print("SURF--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
# -- Draw matches
img_matches = np.empty((max(img_logo.shape[0], img_scene.shape[0]),
                        img_logo.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
# cv.drawMatches(img_logo, keypoints_obj, img_scene, keypoints_scene,
#              good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches = cv.imread('meuble.jpg')
height = int(img_matches.shape[0] * width/img_matches.shape[1])
dim = (width, height)
# resize image
img_matches = cv.resize(img_matches, dim, interpolation=cv.INTER_AREA)
print("DRAW--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
# -- Localize the object
obj = np.empty((len(good_matches), 2), dtype=np.float32)
scene = np.empty((len(good_matches), 2), dtype=np.float32)
for i in range(len(good_matches)):
    # -- Get the keypoints from the good matches
    obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
    obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
    scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
    scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
H, _ = cv.findHomography(obj, scene, cv.RANSAC)

print("LOCALIZE--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
# -- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4, 1, 2), dtype=np.float32)
obj_corners[0, 0, 0] = 0
obj_corners[0, 0, 1] = 0
obj_corners[1, 0, 0] = img_logo.shape[1]
obj_corners[1, 0, 1] = 0
obj_corners[2, 0, 0] = img_logo.shape[1]
obj_corners[2, 0, 1] = img_logo.shape[0]
obj_corners[3, 0, 0] = 0
obj_corners[3, 0, 1] = img_logo.shape[0]
scene_corners = cv.perspectiveTransform(obj_corners, H)
x_cords = [scene_corners[0, 0, 0], scene_corners[1, 0, 0],
           scene_corners[2, 0, 0], scene_corners[3, 0, 0], scene_corners[0, 0, 0]]
y_cords = [scene_corners[0, 0, 1], scene_corners[1, 0, 1],
           scene_corners[2, 0, 1], scene_corners[3, 0, 1], scene_corners[1, 0, 0]]
area_etiquette = 0
for x in range(4-2):
    v1, v2, v3 = 0, x+1, x+2
    tr_area = abs(0.5*(x_cords[v1]*(y_cords[v2]-y_cords[v3]) +
                       x_cords[v2]*(y_cords[v3]-y_cords[v1]) +
                       x_cords[v3]*(y_cords[v1]-y_cords[v2])))
    area_etiquette += tr_area


# -- Draw lines between the corners (the mapped object in the scene - image_2 )
cv.line(img_matches, (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])),
        (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
cv.line(img_matches, (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])),
        (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
cv.line(img_matches, (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])),
        (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
cv.line(img_matches, (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])),
        (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)
# print(
#     scene_corners[0, 0, 0], ",", scene_corners[0, 0, 1], ";",
#     scene_corners[1, 0, 0], ",", scene_corners[1, 0, 1], ";",
#     scene_corners[2, 0, 0], ",", scene_corners[2, 0, 1], ";",
#     scene_corners[3, 0, 0], ",", scene_corners[3, 0, 1]
# )
# -- Show detected matches
print("Draw lines--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
cv.imwrite('res.jpg', img_matches)


###########################################################################
######################## suppression du background ########################
###########################################################################

# -*- coding: utf-8 -*-
img = cv.imread('meuble.jpg')
width = 1000
height = int(img.shape[0] * width/img.shape[1])
dim = (width, height)
# resize image
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
cv.imwrite('resized.jpg', resized)
# fin resized
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iteration = 3
r = 8
print("Remplissage--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mask_fore = np.zeros_like(resized)
mask_back = np.zeros_like(resized)
print("debut back--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
# cv.namedWindow('image')
x, y, _ = resized.shape
cv.line(mask_fore, (r, r), (r, x-r), (0, 255, 0), r)
cv.line(mask_fore, (r, x-r), (y-r, x-r), (0, 255, 0), r)
cv.line(mask_fore, (y-r, x-r), (y-r, r), (0, 255, 0), r)
cv.line(mask_fore, (y-r, r), (r, r), (0, 255, 0), r)
print("lines--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mask_global = np.zeros(resized.shape[:2], np.uint8) + 2
mask_global[mask_fore[:, :, 1] == 255] = 1
print("masks--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mask_global, bgdModel, fgdModel = cv.grabCut(
    resized, mask_global, None, bgdModel, fgdModel, iteration, cv.GC_INIT_WITH_MASK)
mask_global = np.where((mask_global == 2), 1, 0).astype(
    'uint8')  # 0, 1 pour inverser
target = resized*mask_global[:, :, np.newaxis]
#cv.imshow('target', target)
print("back retiré--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
cv.imwrite('result.jpg', target)
img = cv.imread('result.jpg', cv.IMREAD_GRAYSCALE)
area_objet_pixel = cv.countNonZero(img)

# img = cv.imread('mini.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# rect = (r, r, x-r, y-r)
# iteration = 3
# cv.grabCut(img, mask, rect, bgdModel, fgdModel,
#            iteration, cv.GC_INIT_WITH_RECT)

# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img*mask2[:, :, np.newaxis]
# cv.imwrite('t1.jpg', img)


#plt.imshow(img), plt.colorbar(), plt.show()
print("deuxième back retiré--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
kernel = np.ones((5, 5), np.uint8)
closing = cv.morphologyEx(target, cv.MORPH_CLOSE, kernel)
cv.imwrite('resultclosing.jpg', target)
print("close--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()


def midpoint(ptA, ptB):
    return (
        (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# load the image, convert it to grayscale, and blur it slightly
image = cv.imread('result.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (7, 7), 0)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv.Canny(gray, 50, 100)
edged = cv.dilate(edged, None, iterations=1)
im = cv.erode(edged, None, iterations=1)

#imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(im, 0, 255, 0)
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv.contourArea(contour)
    if area > 200000:
        #cv.drawContours(im, contours, -1, (0, 255, 0), 3)
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        print(w, h)
        # draw the book contour (in green)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv.circle(image, (int(x)+int(w), int(y)+int(h)), 5, (0, 0, 255), -1)
im = cv.imwrite('result_contour.jpg', image)

###########################################################################
########################### calcul des contours ###########################
###########################################################################

# # Reading same image in another
# # variable and converting to gray scale.
# font = cv.FONT_HERSHEY_COMPLEX
# img2 = cv.imread('result.jpg', cv.IMREAD_COLOR)
# img = cv.imread('result.jpg', cv.IMREAD_GRAYSCALE)

# # Converting image to a binary image
# # ( black and white only image).
# _, threshold = cv.threshold(img, 110, 255, cv.THRESH_BINARY)

# # Detecting contours in image.
# contours, _= cv.findContours(threshold, cv.RETR_TREE,
#                                cv.CHAIN_APPROX_SIMPLE)

# # Going through every contours found in the image.
# for cnt in contours :

#     approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)

#     # draws boundary of contours.
#     cv.drawContours(img2, [approx], 0, (0, 0, 255), 5)

#     # Used to flatted the array containing
#     # the co-ordinates of the vertices.
#     n = approx.ravel()
#     i = 0

#     for j in n :
#         if(i % 2 == 0):
#             x = n[i]
#             y = n[i + 1]

#             # String containing the co-ordinates.
#             string = str(x) + " " + str(y)

#             if(i == 0):
#                 # text on topmost co-ordinate.
#                 cv.putText(img2, "Arrow tip", (x, y),
#                                 font, 0.5, (255, 0, 0))
#             else:
#                 # text on remaining co-ordinates.
#                 cv.putText(img2, string, (x, y),
#                           font, 0.5, (0, 255, 0))
#         i = i + 1

# # Showing the final image.
# cv.imshow('image2', img2)

# ###########################################################################
# ############################# calcul de l'aire ############################
# ###########################################################################
# img1 = img2
# # Convert it to greyscale
# img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# # Threshold the image
# ret, thresh = cv.threshold(img,50,255,0)
# # Find the contours
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# # For each contour, find the convex hull and draw it
# # on the original image.
# for i in range(len(contours)):
#     hull = cv.convexHull(contours[i],1)
#     cv.drawContours(img1, [hull], -1, (255, 0, 0), 2)
# # Display the final convex hull image
# cv.imshow('ConvexHull', img1)


# a = np.array(img1[:,:,0])
# _, contour= cv.findContours(a,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
# area = cv.contourArea(contours[0])
# # print(area)
# #print(np.count_nonzero(a))
print(
    scene_corners[0, 0, 0], ",", scene_corners[0, 0, 1], ";",
    scene_corners[1, 0, 0], ",", scene_corners[1, 0, 1], ";",
    scene_corners[2, 0, 0], ",", scene_corners[2, 0, 1], ";",
    scene_corners[3, 0, 0], ",", scene_corners[3, 0, 1], ";",
    area_etiquette, ";", area_objet_pixel
)


# while True:
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()
