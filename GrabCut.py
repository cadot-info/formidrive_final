# -*- coding: utf-8 -*-
import numpy as np
import cv2
img = cv2.imread('mini.jpg')
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iteration = 3
r = 8

mask_fore = np.zeros_like(img)
mask_back = np.zeros_like(img)
cv2.namedWindow('image')
cv2.line(mask_fore,(r,r),(r,440-r),(0,255,0),r)
cv2.line(mask_fore,(r,440-r),(590-r,440-r),(0,255,0),r)
cv2.line(mask_fore,(590-r,440-r),(590-r,r),(0,255,0),r)
cv2.line(mask_fore,(590-r,r),(r,r),(0,255,0),r)
mask_global = np.zeros(img.shape[:2], np.uint8) + 2
mask_global[mask_fore[:, :, 1] == 255] = 1
mask_global, bgdModel, fgdModel = cv2.grabCut(img, mask_global, None, bgdModel, fgdModel, iteration, cv2.GC_INIT_WITH_MASK)
mask_global = np.where((mask_global == 2) , 1, 0).astype('uint8') #0, 1 pour inverser
target = img*mask_global[:, :, np.newaxis]
cv2.imshow('target', target)
cv2.imwrite('result.jpg', target)


# Reading same image in another  
# variable and converting to gray scale. 
font = cv2.FONT_HERSHEY_COMPLEX 
img2 = cv2.imread('result.jpg', cv2.IMREAD_COLOR) 
img = cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE) 
  
# Converting image to a binary image 
# ( black and white only image). 
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY) 
  
# Detecting contours in image. 
contours, _= cv2.findContours(threshold, cv2.RETR_TREE, 
                               cv2.CHAIN_APPROX_SIMPLE) 

# Going through every contours found in the image. 
for cnt in contours : 
  
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
  
    # draws boundary of contours. 
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)  
  
    # Used to flatted the array containing 
    # the co-ordinates of the vertices. 
    n = approx.ravel()  
    i = 0
  
    for j in n : 
        if(i % 2 == 0): 
            x = n[i] 
            y = n[i + 1] 
  
            # String containing the co-ordinates. 
            string = str(x) + " " + str(y)  
  
            if(i == 0): 
                # text on topmost co-ordinate. 
                cv2.putText(img2, "Arrow tip", (x, y), 
                                font, 0.5, (255, 0, 0))  
            else: 
                # text on remaining co-ordinates. 
                cv2.putText(img2, string, (x, y),  
                          font, 0.5, (0, 255, 0))  
        i = i + 1
  
# Showing the final image. 
cv2.imshow('image2', img2)  








img1 = cv2.imread('result.jpg')
# Convert it to greyscale
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Threshold the image
ret, thresh = cv2.threshold(img,50,255,0)
# Find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# For each contour, find the convex hull and draw it
# on the original image.
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i],1)
    cv2.drawContours(img1, [hull], -1, (255, 0, 0), 2)
# Display the final convex hull image
cv2.imshow('ConvexHull', img1)


a = np.array(img1[:,:,0])
_, contour= cv2.findContours(a,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
area = cv2.contourArea(contours[0])
print(area)
print(np.count_nonzero(a))






while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
