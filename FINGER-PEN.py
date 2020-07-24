import cv2
import numpy as np
import draw_rect
import hand_histogram
import hist_masking
import manage_image_opr
global traverse_point
traverse_point=[]
def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
global hand_hist
is_hand_hist_created = False
capture = cv2.VideoCapture(1)
while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if pressed_key & 0xFF == ord(' '):
            is_hand_hist_created = True
            hand_hist = hand_histogram.hand_histogram(frame,hand_rect_one_x,hand_rect_one_y,hand_rect_two_x,hand_rect_two_y)

        if is_hand_hist_created:
            frame,traverse_point=manage_image_opr.manage_image_opr(frame, hand_hist,traverse_point)

        else:
            frame,hand_rect_one_x,hand_rect_one_y,hand_rect_two_x,hand_rect_two_y = draw_rect.draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

cv2.destroyAllWindows()
capture.release()