import cv2
import numpy as np
import argparse
import time
from finger_detect.finger_calibration import real_finger_calib, manage_image_opr, coor_key_transform
from kb_detect.img_calibration import ref_keyboard_calib
from kb_detect.vdo_calibration import real_keyboard_calib

def real_test(opt):
    capture = cv2.VideoCapture(1)
    last_time = time.time()  

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        # frame = cv2.flip(frame, 1)
        
        hand_hist = np.load(opt.finger_hist_path)
        coor = manage_image_opr(frame, hand_hist)
        current_time = time.time()
        if coor and current_time - last_time >= 1:
            coor_key_transform(opt, coor)
            last_time = current_time
        cv2.imshow("Keyboard operator detection", frame)

        if pressed_key == 27: #ESC
            break 
    cv2.destroyAllWindows()
    capture.release()

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
    parser.add_argument('-ref_key_json_path', help='the path of saving each key boundary location of the reference keyboard', type=str, default='./json_file/keyboard_ref.json')
    # Finger calibration
    parser.add_argument('-finger_hist_path', help='the path of saving finger hist', type=str, default='./finger_detect/hand_hist.npy')
    # Real keyboard calibration
    parser.add_argument('-real_key_json_path', help='the path of saving real env keyboard location', type=str, default='./json_file/keyboard_real.json')
    # configuration
    parser.add_argument('-mode', choices=['refKeyCalib', 'realFingCalib', 'realKeyCalib', 'realTest'], help='choose the stage to excuate', required=True, type=str)
    opt = parser.parse_args()

    main(opt)