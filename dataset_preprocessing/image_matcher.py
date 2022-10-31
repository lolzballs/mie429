import numpy as np 
import cv2
from matplotlib import pyplot as plt
import statistics
import pandas as pd 
from scipy import stats

class ImageMatcher:
    def __init__(self, baseImage):
        self.baseImg = baseImage
        self.searchImg = []
        self.refineData, self.drawZone = False, False
        self.allignmentStage = 0


    def uploadSearchImage(self, searchImage):
        if searchImage.any() == None:
            print("image could not be found")
            return False
        self.searchImg = searchImage.copy()
        return True 

    def findObject(self, paddingX=0, paddingY=0, draw_box=False):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.baseImg,None)
        kp2, des2 = orb.detectAndCompute(self.searchImg,None)
    
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        good_matches = matches[:10]
        
        pointsX = []
        pointsY = []
        for m in good_matches:
            pointsX.append(kp2[m.trainIdx].pt[0])
            pointsY.append(kp2[m.trainIdx].pt[1])

        kp_img = cv2.drawMatches(self.baseImg,kp1,self.searchImg,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.searchImg, mask = self.__getBoundingBox(pointsX, pointsY, self.searchImg, paddingX, paddingY, draw_box)
        
        if mask is not None:
            self.__fill_box(mask)

        return kp_img, self.searchImg
    
    def __fill_box(self, mask):
        est_bkgd_color = np.percentile(self.searchImg, 30)
        self.searchImg = np.multiply(self.searchImg, mask)
        mask[mask < 1] = est_bkgd_color
        self.searchImg = np.add(self.searchImg, mask)

    def __getBoundingBox(self, dataX, dataY, img, paddingX, paddingY, draw_box):
        minX = max(0, int(min(dataX) - paddingX/2))
        minY = max(0, int(min(dataY) - paddingY/2))
        maxX = min(img.shape[1], int(max(dataX) + paddingX/2))
        maxY = min(img.shape[0], int(max(dataY) + paddingY/2))
        bounding_box_area = (maxX-minX)*(maxY-minY)
        img_area = img.shape[0]*img.shape[1]
        
        # Logic check, if box is more than 10% of the image its probably a bad detection
        bad_detection = bounding_box_area > 0.1*img_area
        if draw_box and not bad_detection:
            cv2.rectangle(img, (minX, minY), (maxX, maxY), (255, 255, 0), 10)
        mask = cv2.rectangle(np.ones_like(img), (minX, minY), (maxX, maxY), (0, 0, 0), -1)

        if bad_detection:
            return img, None
        return img, mask
        
