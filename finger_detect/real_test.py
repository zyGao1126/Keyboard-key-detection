import numpy as np
import cv2
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb_detect.ini_calibration import processImage
from kb_detect.real_calibration import get_contour
from finger_detect.finger_calibration import manage_image_opr, coor_key_transform


def matrixChange(config, last_matrix, matrix):
    if last_matrix is None:
        return False
    diff_sita = np.abs(np.arccos(np.clip(last_matrix[0][0], -1, 1)) - np.arccos(np.clip(matrix[0][0], -1, 1)))
    diff_x = np.abs(last_matrix[0][2] - matrix[0][2])
    diff_y = np.abs(last_matrix[1][2] - matrix[1][2])
    # print("diff_sita: {}  diff_x: {}  diff_y: {}".format(diff_sita, diff_x, diff_y))
    if diff_sita >= config['sita_threshold'] or diff_x >= config['coor_threshold'] or diff_y >= config['coor_threshold']:
        return True
    return False

def real_test(config):
    config_refKey = config['refKeyCalib']
    config_finger = config['realFingerCalib']
    config_realKey = config['realKeyCalib']
    config_test = config['realTest']
    
    capture = cv2.VideoCapture(1)  
    ref_area = last_area = None
    ref_matrix = last_matrix = None
    count = 0

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        # frame = cv2.flip(frame, 1)

        with open("./json_file/limit_real.json") as f:
            ranges = (json.load(f))["limits"]        
        binary_image = cv2.cvtColor(processImage(ranges, frame), cv2.COLOR_BGR2GRAY)  
        _, _, matrix, area = get_contour(config_realKey, binary_image, frame, config_refKey['ref_key_json_path'])
        if area is not None:
            if ref_area is None:
                if count < 5:
                    count = count + 1 if not matrixChange(config_test, last_matrix, matrix) else 0
                else:
                    ref_area = last_area 
                    ref_matrix = last_matrix
                last_area = area
                last_matrix = matrix 
            else:
                area_th = ref_area * 0.05
                if abs(area - ref_area) < area_th and matrixChange(config_test, ref_matrix, matrix):
                    ref_area = ref_matrix = None
                    last_area = area
                    last_matrix = matrix 
                    count = 0
        hand_hist = np.load(config_finger['finger_hist_path'])
        coor = manage_image_opr(frame, hand_hist, config_test['coor_bias'])
        if coor and ref_matrix is not None:
            coor_key_transform(config_refKey['ref_key_json_path'], coor, ref_matrix)
        cv2.imshow("Keyboard operator detection", frame)

        if pressed_key == 27: #ESC
            break 
    cv2.destroyAllWindows()
    capture.release()