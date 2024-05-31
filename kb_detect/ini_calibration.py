#
# python kb_detect/img_calibration 
#

import os
import sys
import numpy as np
import cv2
from functools import partial
import copy
import json
from colorama import Fore, Back, Style
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb_detect.util import *

GREEN = (0,255,0)

def processImage(ranges, image):

    # processing
    mins = np.array([ranges['B']['min'], ranges['G']['min'], ranges['R']['min']])
    maxs = np.array([ranges['B']['max'], ranges['G']['max'], ranges['R']['max']])

    # mask
    mask = cv2.inRange(image, mins, maxs)
    # conversion from numpy from uint8 to bool
    mask = mask.astype(bool)

    # process the image
    image_processed = copy.deepcopy(image)
    image_processed[np.logical_not(mask)] = 0

    # get binary image with threshold the values not in the mask
    _, image_processed = cv2.threshold(image_processed, 1, 255, cv2.THRESH_BINARY)
    # image_processed = cv2.bitwise_not(image_processed)

    return image_processed

def onTrackbar(value, channel, min_max, ranges):
    print("Selected threshold "+ Style.BRIGHT + Fore.YELLOW + str(value) + Style.RESET_ALL + " for limit " + Style.BRIGHT + Fore.GREEN + channel + min_max + Style.RESET_ALL)

    # update range values
    ranges[channel][min_max] = value

def get_RGB_range(rootPath, json_path):
    window_name_cal = 'calibration'
    window_name_or = 'Original'
    cv2.namedWindow(window_name_cal, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_or, cv2.WINDOW_AUTOSIZE)    

    ranges = {'B': {'max': 255, 'min': 50},
              'G': {'max': 255, 'min': 0},
              'R': {'max': 255, 'min': 0}}

    cv2.createTrackbar('Rmin', window_name_cal, ranges["R"]["min"], 255, partial(onTrackbar, channel='R', min_max='min', ranges=ranges))
    cv2.createTrackbar('Rmax', window_name_cal, ranges["R"]["max"], 255, partial(onTrackbar, channel='R', min_max='max', ranges=ranges))
    cv2.createTrackbar('Gmin', window_name_cal, ranges["G"]["min"], 255, partial(onTrackbar, channel='G', min_max='min', ranges=ranges))
    cv2.createTrackbar('Gmax', window_name_cal, ranges["G"]["max"], 255, partial(onTrackbar, channel='G', min_max='max', ranges=ranges))
    cv2.createTrackbar('Bmin', window_name_cal, ranges["B"]["min"], 255, partial(onTrackbar, channel='B', min_max='min', ranges=ranges))
    cv2.createTrackbar('Bmax', window_name_cal, ranges["B"]["max"], 255, partial(onTrackbar, channel='B', min_max='max', ranges=ranges))    

    image = cv2.imread(rootPath)

    while True:
        k = cv2.waitKey(1)
        if k == ord("q"):
            break    
        
        # process image
        processed_image = processImage(ranges, image)
        cv2.imshow(window_name_cal, processed_image)
        cv2.imshow(window_name_or, image)

        if k == ord("w"):
            write_json(json_path, ranges)
            print('writing image color limits to file ' + Style.BRIGHT + Fore.GREEN + json_path + Style.RESET_ALL)
            break

    cv2.destroyAllWindows()

def get_binary_image(limit_file, image):
    with open(limit_file) as f:
        ranges = (json.load(f))["limits"]

    return processImage(ranges, image)

def get_key_centroid(image, th_area, visual=True):
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), connectivity, cv2.CV_32S)
    height, width = image.shape[0:2]

    # get labels by max area value
    max_labels = sorted([(i, stats[i, cv2.CC_STAT_AREA], [(stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3])]) for i in range(0, nb_components)], key=lambda x: x[1], reverse=True)    

    objects = []
    for idx, label in enumerate(max_labels):
        # adjust based on the generated centroid 
        # if label[0] != 0 and (label[0] <= 1 or label[1] < th_area or idx <= 2):
        if label[0] == 1 or label[1] < th_area:
            continue 
        # draw the bounding rectangele around each object
        cv2.rectangle(image, label[2][0], label[2][1], GREEN, 2)
        objects.append(label)        

    if visual:
        cv2.namedWindow("Centroid image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Centroid image', image.shape[1], image.shape[0])  
        cv2.imshow('Centroid image', image)           
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return objects     

def anno_key_location(event, x, y, flags, param, keyboard):      
    if event == cv2.EVENT_LBUTTONDOWN:
        # for the handle keyboard function
        print(x, y)       
        letter = input(f"What should i save ({x}, {y}) as? ") 
        if letter:
            keyboard[letter] = (x,y)    
    
def write_key_2_json(kb_file, keyboard, centroids):
    real_keyboard = {}
    # pre-write the whole keyboard boundary
    real_keyboard["keyboard"] = [int(centroids[0][2][0][0]), int(centroids[0][2][0][1]), int(centroids[0][2][1][0]), int(centroids[0][2][1][1])]
    del centroids[0]
    for key, value in keyboard.items():
        (x,y) = value
        for label in centroids:
            if x > label[2][0][0] and y > label[2][0][1]:
                if x < label[2][1][0] and y < label[2][1][1]:
                    # double check
                    real_keyboard[key] = [int(label[2][0][0]), int(label[2][0][1]), int(label[2][1][0]), int(label[2][1][1])]

    formatted_keyboard = {key: f"({value[0]}, {value[1]}, {value[2]}, {value[3]})" for key, value in real_keyboard.items()}
    with open(kb_file, 'w') as file:
        json.dump(formatted_keyboard, file, indent=4)    

def mouse_handler_test(event, x, y, flags, params, keyboard):
    if event == cv2.EVENT_LBUTTONDOWN:
        flag1 = flag2 =False
        for key, label in keyboard.items():            
            (x0, y0, x1, y1) = eval(label)
            if x > x0 and y > y0:
                if x < x1 and y < y1:
                    if key == 'keyboard':
                        flag1 = True
                        print("Point in keyboard, ", end="")
                    else:
                        flag2 = True
                        print("the key is: ", key)
        if not flag1 and not flag2:
            print("Point is not in keyboard.")
        elif flag1 and not flag2:
            print("but not belong to any key.")

def key_annotation(image, image_processed, key_json_path, centroids):
    keyboard = {}
    cv2.namedWindow("key annotation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('key annotation', image.shape[1], image.shape[0])        
    cv2.setMouseCallback("key annotation", partial(anno_key_location, keyboard=keyboard)) 
    while True:
        cv2.imshow('key annotation', image_processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if len(keyboard):
        write_key_2_json(key_json_path, keyboard, centroids)
    cv2.destroyAllWindows()  

def test_key_annotation(image_processed, key_json_path):
    real_keyboard = {}
    with open(key_json_path) as f:
        real_keyboard = json.load(f)
    cv2.namedWindow("annotation test")    
    cv2.setMouseCallback("annotation test", partial(mouse_handler_test, keyboard=real_keyboard))
    while True:
        cv2.imshow('annotation test', image_processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()   

def ref_keyboard_calib(config):
    img_json_path = './json_file/limits_ref.json'
    key_json_path = config['ref_key_json_path']
    img_path = config['ref_img_path']
    threshold_area = config['th_key_area']
    
    # step1: get color range according to the mode
    get_RGB_range(img_path, img_json_path)
    # step2: use color range to get binary image
    image = cv2.imread(img_path)
    image_processed = get_binary_image(img_json_path, image)
    # step3: get each key's centroid and boundary
    centroids = get_key_centroid(image_processed, threshold_area)
    # step4: annotate keyboard key boundary to .json
    key_annotation(image, image_processed, key_json_path, centroids)
    # step5: test annotation
    test_key_annotation(image_processed, key_json_path)

if __name__ == '__main__':
    img_path = './iniImage/keyboard_ref.jpg'
    img_json_path = './json_file/limits_ref.json'
    key_json_path = './json_file/keyboard_ref.json'
    visual = True  
    need_annotate = False  
    keyboard = {}
    real_keyboard = {}
    threshold_area = 180

    get_RGB_range(img_path, img_json_path)
    image = cv2.imread(img_path)
    image_processed = get_binary_image(img_json_path, image)    
    centroids = get_key_centroid(image_processed, threshold_area, visual)
    if need_annotate:
        key_annotation(image, image_processed, key_json_path)
    test_key_annotation(image_processed, key_json_path)
    


