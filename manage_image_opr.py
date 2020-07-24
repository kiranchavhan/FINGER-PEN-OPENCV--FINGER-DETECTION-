import cv2
import numpy as np
import hist_masking
from matplotlib import *
# reference:-https://github.com/amarlearning/Finger-Detection-and-Tracking
def threshold(mask):
    """Thresholding into a binary mask"""
    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayMask, 0, 255, 0)
    return thresh
def getMaxContours(contours):
    """Find the largest contour"""
    maxIndex = 0
    maxArea = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxIndex = i
    return contours[maxIndex]
def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None
	
def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None
def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
           cv2.line(img=frame, pt1=traverse_point[i], pt2=traverse_point[i], color=(255, 0, 0), thickness=20, lineType=8, shift=0)
    return frame
def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def manage_image_opr(frame, hand_hist,traverse_point):
    hist_mask_image = hist_masking.hist_masking(frame, hand_hist)
    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)
    contour_list = contours(hist_mask_image)
    max_cont = max(contour_list, key=cv2.contourArea)
 
    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)
    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        traverse_point.append(far_point)

        return draw_circles(frame, traverse_point),traverse_point
