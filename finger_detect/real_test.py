import numpy as np
import cv2
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb_detect.ini_calibration import processImage
from kb_detect.real_calibration import get_contour
from finger_detect.finger_calibration import manage_image_opr, coor_key_transform


def matrixChange(last_matrix, matrix):
    if last_matrix is None:
        return False
    diff_sita = np.abs(np.arccos(np.clip(last_matrix[0][0], -1, 1)) - np.arccos(np.clip(matrix[0][0], -1, 1)))
    diff_x = np.abs(last_matrix[0][2] - matrix[0][2])
    diff_y = np.abs(last_matrix[1][2] - matrix[1][2])
    print("diff_sita: {}  diff_x: {}  diff_y: {}".format(diff_sita, diff_x, diff_y))
    if diff_sita >= 2 or diff_x >= 10 or diff_y >= 10:
        return True
    return False

def real_test(opt):
    capture = cv2.VideoCapture(1)  
    ref_area = last_area = None
    ref_matrix = last_matrix = None
    count = 0
    area_th = 1000


    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        # frame = cv2.flip(frame, 1)

        with open("./json_file/limit_real.json") as f:
            ranges = (json.load(f))["limits"]        
        binary_image = cv2.cvtColor(processImage(ranges, frame), cv2.COLOR_BGR2GRAY)  
        edge_image, warped_image, matrix, area = get_contour(binary_image, frame)
        if area is not None:
            if ref_area is None:
                if count < 3:
                    count = count + 1 if not matrixChange(last_matrix, matrix) else 0
                else:
                    ref_area = last_area 
                    ref_matrix = last_matrix
                last_area = area
                last_matrix = matrix 
            else:
                if abs(area - ref_area) < area_th and matrixChange(ref_matrix, matrix):
                    ref_area = ref_matrix = None
                    last_area = area
                    last_matrix = matrix 
                    count = 0
        hand_hist = np.load(opt.finger_hist_path)
        coor = manage_image_opr(frame, hand_hist)
        if coor and ref_matrix is not None:
            coor_key_transform(opt, coor, ref_matrix)
        cv2.imshow("Keyboard operator detection", frame)

        if pressed_key == 27: #ESC
            break 
    cv2.destroyAllWindows()
    capture.release()