import numpy as np 
import cv2

class ImageMatcher:
    def __init__(self, base_image):
        self.base_img = base_image
        self.search_img = []
        self.refine_data, self.draw_zone = False, False
        self.allignment_stage = 0


    def upload_search_image(self, search_image):
        if search_image.any() == None:
            print("image could not be found")
            return False
        self.search_img = search_image.copy()
        return True 

    def find_object(self, padding_X=0, padding_Y=0, draw_box=False):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.base_img, None)
        kp2, des2 = orb.detectAndCompute(self.search_img, None)
        if len(kp2)==0 or des2 is None:
            return None, None
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        good_matches = matches[:10]
        
        points_X = []
        points_Y = []
        for m in good_matches:
            points_X.append(kp2[m.trainIdx].pt[0])
            points_Y.append(kp2[m.trainIdx].pt[1])

        kp_img = cv2.drawMatches(self.base_img, kp1, self.search_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.search_img, mask = self.__get_bounding_box(points_X, points_Y, self.search_img, padding_X, padding_Y, draw_box)
        
        if mask is not None:
            self.__fill_box(mask)

        return kp_img, self.search_img
    
    def __fill_box(self, mask):
        est_bkgd_color = np.percentile(self.search_img, 30)
        self.search_img = np.multiply(self.search_img, mask)
        mask[mask < 1] = est_bkgd_color
        self.search_img = np.add(self.search_img, mask)

    def __get_bounding_box(self, data_X, data_Y, img, padding_X, padding_Y, draw_box):
        min_X = max(0, int(min(data_X) - padding_X/2))
        min_Y = max(0, int(min(data_Y) - padding_Y/2))
        max_X = min(img.shape[1], int(max(data_X) + padding_X/2))
        max_Y = min(img.shape[0], int(max(data_Y) + padding_Y/2))
        bounding_box_area = (max_X-min_X)*(max_Y-min_Y)
        img_area = img.shape[0]*img.shape[1]
        
        # Logic check, if box is more than 10% of the image its probably a bad detection
        bad_detection = bounding_box_area > 0.1*img_area
        if draw_box and not bad_detection:
            cv2.rectangle(img, (min_X, min_Y), (max_X, max_Y), (255, 255, 0), 10)
        mask = cv2.rectangle(np.ones_like(img), (min_X, min_Y), (max_X, max_Y), (0, 0, 0), -1)

        if bad_detection:
            return img, None
        return img, mask
        
