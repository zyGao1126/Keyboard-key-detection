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

def load_config(config_file_path):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    return config

def main(config, mode):

    if mode == 'refKeyCalib':
        ref_keyboard_calib(config['refKeyCalib'])
    elif mode == 'realFingCalib':
        real_finger_calib(config['realFingerCalib'])
    elif mode == 'realKeyCalib':
        real_keyboard_calib(config['realKeyCalib'], config['refKeyCalib']['ref_key_json_path'])
    elif mode == 'realTest':
        real_test(config)
    else:
        print("Please choose CORRECT mode.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter configuration')
    parser.add_argument('-mode', choices=['refKeyCalib', 'realFingCalib', 'realKeyCalib', 'realTest'], default='realTest', help='choose the stage to excuate', required=True, type=str)
    opt = parser.parse_args()

    config_file_path = "./config.yaml"
    config = load_config(config_file_path)
    main(config, opt.mode)