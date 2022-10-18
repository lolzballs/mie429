import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

from image_matcher import ImageMatcher

# Masks by setting all non-255 values to 0
def absolute_max(img):
    img_cp = img.copy()
    img_cp[img_cp < 255] = 0
    return img_cp

# Floors and rounds based on a percentile threshold value
def threshold_max(img, percentile=99):
    img_cp = img.copy()
    threshold = np.percentile(img, percentile)
    img_cp[img_cp < threshold] = 0
    img_cp[img_cp > 0] = 255
    return img_cp

# Template Matching
def brute_force_temp_matching(img):
    icon = cv.imread('icon.png',cv.IMREAD_GRAYSCALE)
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(icon,None)
    kp2, des2 = orb.detectAndCompute(img,None)
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    kp_img = cv.drawMatches(icon,kp1,img,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return kp_img

# Combines template matching + threshold maxing + draws a circular bounding box
def image_matcher_class(img):
    imgMatcher = ImageMatcher(cv.imread('icon.png'))
    # img = threshold_max(img)
    # cv.GaussianBlur(img,(5,5),0)
    imgMatcher.uploadSearchImage(img)
    return imgMatcher.findObject(paddingX=125, paddingY=175, draw_box=True)
    

dir = "./sample_data"
files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

start_file = 4967

# Other alternatives broken right now due to different return contents
alternatives = [
                # absolute_max,
                # threshold_max,
                # brute_force_temp_matching,
                image_matcher_class
                ]

for alt in alternatives:
    for i in range(0, 10):
        print(f"Testing with file {files[i]}")
        img = cv.imread(f"./sample_data/{files[i]}")
        img_cp = img.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)	
        result_img = alt(gray_img)
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(result_img)
        plt.show()
