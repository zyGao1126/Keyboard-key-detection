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

def get_vertex(image, eps, area_threshold):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if idx >= 5:
            return None, None        
        if area < area_threshold[0] or area > area_threshold[1]:
            continue
        
        epsilon = eps * cv2.arcLength(contour, True) 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2), area

    return None, None


def image_transform(edge_image, image, eps, area_threshold, keybaord_json):
    corners, area = get_vertex(edge_image, eps, area_threshold)    
    if corners is not None:
        cv2.polylines(image, [corners], isClosed=True, color=(0,255,0), thickness=2)    #Green    
        
        s = corners.sum(axis = 1)
        diff = np.diff(corners, axis=1)
        rect = np.array([corners[np.argmin(s)], corners[np.argmin(diff)], corners[np.argmax(s)], corners[np.argmax(diff)]], dtype="float32")
        
        with open(keybaord_json) as f:
            vertexs = eval((json.load(f))["keyboard"])
        w = vertexs[2] - vertexs[0]
        h = vertexs[3] - vertexs[1]
        pts_dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')

        matrix = cv2.getPerspectiveTransform(rect, pts_dst)
        warped_image = cv2.warpPerspective(image, matrix, (w, h))
        return warped_image, matrix, area
    return None, None, None


def edge_detect(image, threshold):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, threshold[0], threshold[1])
    return edged


def get_contour(config, binary_image, image, keyboard_json):
    gray_ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enh_image = morpholo_process(cv2.bitwise_not(binary_image), config['erode_kernel'])    
    # enh_image = cv2.bitwise_not(binary_image)
    
    edge_image = edge_detect(gray_ori_image, config['canny_threshold'])
    diff_image = edge_image - enh_image
    warped_image, matrix, area = image_transform(diff_image, image, config['epsilon'], config['contour_area'], keyboard_json)
    return diff_image, warped_image, matrix, area


def morpholo_process(image, kernel):
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    eroded_image = cv2.erode(image, erode_kernel, iterations=1)
    return eroded_image


def real_keyboard_calib(config, keyboard_json):    
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
        edge_image, warped_image, _, _ = get_contour(config, binary_image, image, keyboard_json)

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
