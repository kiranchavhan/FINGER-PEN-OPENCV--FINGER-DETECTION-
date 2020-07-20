import cv2
import numpy
import pandas
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
