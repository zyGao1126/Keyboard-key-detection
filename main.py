import cv2
import numpy as np
import argparse
import json
import yaml
from collections import deque
from finger_detect.finger_calibration import real_finger_calib
from kb_detect.ini_calibration import ref_keyboard_calib
from kb_detect.real_calibration import real_keyboard_calib
from finger_detect.real_test import real_test

def main(opt):
    if opt.mode == 'refKeyCalib':
        ref_keyboard_calib(opt)
    elif opt.mode == 'realFingCalib':
        real_finger_calib(opt)
    elif opt.mode == 'realKeyCalib':
        real_keyboard_calib(opt)
    elif opt.mode == 'realTest':
        real_test(opt)
    else:
        print("Please choose CORRECT mode.")
        exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter configuration')
    # Reference keyboard registation
    parser.add_argument('-ref_img_path', help='the reference keyboard image path', type=str, default='./iniImage/keyboard_ref.jpg')
    parser.add_argument('-key_area', help='the min area of each key, use as a threshold', type=int, default=180)
    parser.add_argument('-ref_key_json_path', help='the path of saving each key boundary location of the reference keyboard', type=str, default='./json_file/keyboard1_ref.json')
    # Finger calibration
    parser.add_argument('-finger_hist_path', help='the path of saving finger hist', type=str, default='./finger_detect/hand_hist.npy')
    # configuration
    parser.add_argument('-mode', choices=['refKeyCalib', 'realFingCalib', 'realKeyCalib', 'realTest'], help='choose the stage to excuate', required=True, type=str)
    opt = parser.parse_args()

    main(opt)