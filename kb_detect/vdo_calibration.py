#
# python kb_detect/vdo_calibration 
#

import numpy as np
import os
import json
import sys 
import cv2
from functools import partial
from colorama import Fore, Back, Style

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb_detect.util import write_json
from kb_detect.img_calibration import onTrackbar, processImage

GREEN = (0,255,0)

real_key_json_path = "./json_file/keyboard_real.json"


def extract_keyboard(frame, processed_frame):
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY), connectivity, cv2.CV_32S)
    max_labels = sorted([(i, stats[i, cv2.CC_STAT_AREA], [(stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3])]) for i in range(0, nb_components)], key=lambda x: x[1], reverse=True)        
    if len(max_labels) <= 1:
        return None, None, None, None
    cv2.rectangle(frame, max_labels[1][2][0], max_labels[1][2][1], GREEN, 2)
    return max_labels[1][2][0][0], max_labels[1][2][0][1], max_labels[1][2][1][0], max_labels[1][2][1][1]

def real_keyboard_calib(opt):    
    window_name_cal = 'Calibration'
    window_name_or = 'Original'
    cv2.namedWindow(window_name_cal, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_or, cv2.WINDOW_AUTOSIZE)    

    ranges = {'B': {'max': 255, 'min': 160},
              'G': {'max': 255, 'min': 0},
              'R': {'max': 255, 'min': 0}}

    cv2.createTrackbar('Rmin', window_name_cal, ranges["R"]["min"], 255, partial(onTrackbar, channel='R', min_max='min', ranges=ranges))
    cv2.createTrackbar('Rmax', window_name_cal, ranges["R"]["max"], 255, partial(onTrackbar, channel='R', min_max='max', ranges=ranges))
    cv2.createTrackbar('Gmin', window_name_cal, ranges["G"]["min"], 255, partial(onTrackbar, channel='G', min_max='min', ranges=ranges))
    cv2.createTrackbar('Gmax', window_name_cal, ranges["G"]["max"], 255, partial(onTrackbar, channel='G', min_max='max', ranges=ranges))
    cv2.createTrackbar('Bmin', window_name_cal, ranges["B"]["min"], 255, partial(onTrackbar, channel='B', min_max='min', ranges=ranges))
    cv2.createTrackbar('Bmax', window_name_cal, ranges["B"]["max"], 255, partial(onTrackbar, channel='B', min_max='max', ranges=ranges))    
    
    capture = cv2.VideoCapture(1)
    while capture.isOpened():
        k = cv2.waitKey(1)
        _, image = capture.read()
        # image = cv2.flip(image, 1)

        # process image
        processed_image = processImage(ranges, image)
        cv2.imshow(window_name_cal, processed_image)
        x1, y1, x2, y2 = extract_keyboard(image, processed_image)
        cv2.imshow(window_name_or, image)    

        if k == ord("w"):
            # vdo_json_path = "./json_file/keyboard_real.json"
            # write_json(vdo_json_path, ranges)
            # print('writing video color limits to file ' + Style.BRIGHT + Fore.GREEN + vdo_json_path + Style.RESET_ALL)
            formatted_keyboard = {"boundary": f"({x1}, {y1}, {x2}, {y2})"}
            with open(opt.real_key_json_path, 'w') as file_handle:
                json.dump(formatted_keyboard, file_handle)
            print('writing current keyboard location to file ' + opt.real_key_json_path)
            break                    

        if k == 27:
            break
    cv2.destroyAllWindows()
