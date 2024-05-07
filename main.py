import cv2
import numpy as np
from finger_detect.finger_calibration import *

hand_hist = None

def main():
    global hand_hist
    capture = cv2.VideoCapture(1)
    is_hand_hist_created = False

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        
        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            print("finger histogram registeration")
            hand_hist = hand_histogram(frame)
        
        if is_hand_hist_created:
            coor = manage_image_opr(frame, hand_hist)
            if coor:
                coor_key_transform(frame, coor)

        else:
            frame = draw_rect(frame)
        
        cv2.imshow("Keyboard operator detection", frame)

        if pressed_key == 27: #ESC
            break
    
    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()