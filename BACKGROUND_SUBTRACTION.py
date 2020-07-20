import cv2
import numpy as np
cap=cv2.VideoCapture(1)
fg=cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame=cap.read()
    fgmask=fg.apply(frame)

    cv2.imshow('original',frame)
    cv2.imshow('fg',fgmask)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()