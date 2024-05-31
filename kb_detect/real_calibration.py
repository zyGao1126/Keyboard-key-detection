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
from kb_detect.ini_calibration import onTrackbar, processImage

GREEN = (0,255,0)

def get_vertex(image):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for idx, contour in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(contour, True) 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if idx > 5:
            return None, None
        elif cv2.contourArea(contour) < 5000 or cv2.contourArea(contour) > 120000:
            continue   
        if len(approx) == 4:
            print(cv2.contourArea(contour))
            return approx.reshape(4, 2), cv2.contourArea(contour)

    return None, None


def image_transform(edge_image, image):
    corners, area = get_vertex(edge_image)    
    if corners is not None:
        cv2.polylines(image, [corners], isClosed=True, color=GREEN, thickness=2)        
        
        h, w = 150, 300
        s = corners.sum(axis = 1)
        diff = np.diff(corners, axis=1)
        rect = np.array([corners[np.argmin(s)], corners[np.argmin(diff)], corners[np.argmax(s)], corners[np.argmax(diff)]], dtype="float32")
        pts_dst = np.array([[0, 0], [w - 1, 0], [w- 1, h - 1], [0, h - 1]], dtype='float32')

        matrix = cv2.getPerspectiveTransform(rect, pts_dst)
        warped_image = cv2.warpPerspective(image, matrix, (w, h))
        return warped_image, matrix, area
    return None, None, None


def edge_detect(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged


def get_contour(binary_image, image):
    gray_ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enh_image = cv2.bitwise_not(morpholo_process(binary_image))    
    # enh_image = cv2.bitwise_not(binary_image)
    
    edge_image = edge_detect(gray_ori_image)
    diff_image = edge_image - enh_image
    warped_image, matrix, area = image_transform(diff_image, image)
    return diff_image, warped_image, matrix, area


def morpholo_process(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    # opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed


def real_keyboard_calib(opt):    
    ori_img_window = 'original_img'
    binary_img_window = 'binary_img'
    calibrate_img_window = 'calibrate_img'
    affine_img_window = 'affine_img'

    ranges = {'B': {'max': 117, 'min': 0},
              'G': {'max': 104, 'min': 0},
              'R': {'max': 255, 'min': 0}}

    cv2.namedWindow(binary_img_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(calibrate_img_window, cv2.WINDOW_AUTOSIZE)     
    cv2.namedWindow(affine_img_window, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('Rmin', binary_img_window, ranges["R"]["min"], 255, partial(onTrackbar, channel='R', min_max='min', ranges=ranges))
    cv2.createTrackbar('Rmax', binary_img_window, ranges["R"]["max"], 255, partial(onTrackbar, channel='R', min_max='max', ranges=ranges))
    cv2.createTrackbar('Gmin', binary_img_window, ranges["G"]["min"], 255, partial(onTrackbar, channel='G', min_max='min', ranges=ranges))
    cv2.createTrackbar('Gmax', binary_img_window, ranges["G"]["max"], 255, partial(onTrackbar, channel='G', min_max='max', ranges=ranges))
    cv2.createTrackbar('Bmin', binary_img_window, ranges["B"]["min"], 255, partial(onTrackbar, channel='B', min_max='min', ranges=ranges))
    cv2.createTrackbar('Bmax', binary_img_window, ranges["B"]["max"], 255, partial(onTrackbar, channel='B', min_max='max', ranges=ranges))

    capture = cv2.VideoCapture(1)
    while capture.isOpened():
        k = cv2.waitKey(1)
        _, image = capture.read()
        # image = cv2.flip(image, 1)

        binary_image = cv2.cvtColor(processImage(ranges, image), cv2.COLOR_BGR2GRAY)  
        edge_image, warped_image, _, _ = get_contour(binary_image, image)

        cv2.imshow(ori_img_window, image)
        cv2.imshow(binary_img_window, binary_image)      
        cv2.imshow(calibrate_img_window, edge_image)   
        if warped_image is not None:
            cv2.imshow(affine_img_window, warped_image)                               

        if k == ord("w"):
            vdo_json_path = "./json_file/limit_real.json"
            write_json(vdo_json_path, ranges)
            print('writing video color limits to file ' + Style.BRIGHT + Fore.GREEN + vdo_json_path + Style.RESET_ALL)
            break                    

        if k == 27:
            break
    cv2.destroyAllWindows()
