import os
import sys
import cv2
import json
import numpy as np


def whole_image_histogram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hand_hist = cv2.calcHist([hsv_frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def sorted_defect_points(defects, contour, centroid):
    point_list = []
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float64)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float64)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        for i in range(len(s)):
            point = tuple(contour[s[i]][0])
            distance = dist[i]
            point_list.append((point, distance))

        point_list.sort(key=lambda x : x[1], reverse=True)  
        return point_list 
    
    return None


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 0.6)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    real_area_ratio = np.sum(thresh == 255) / thresh.size
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh), real_area_ratio


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def manage_image_opr(frame, hand_hist, real_area_threshold):
    hist_mask_image, area_ratio = hist_masking(frame, hand_hist)
    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image)
    if area_ratio < real_area_threshold or len(contour_list) < 1:
        return None
    max_cont = max(contour_list, key=cv2.contourArea)
    cnt_centroid = centroid(max_cont)

    hull = cv2.convexHull(max_cont, returnPoints=False)
    defects = cv2.convexityDefects(max_cont, hull)
    sorted_points = sorted_defect_points(defects, max_cont, cnt_centroid)
    
    return sorted_points


def is_in_keyboard(width, height, point):
    if point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height:
        return False
    else:
        return True


def point_transform(keypoint, matrix):
    keypoint = np.array([keypoint[0], keypoint[1]]).reshape(-1, 1, 2)
    affine_keypoint = cv2.perspectiveTransform(keypoint.astype('float32'), matrix)
    affine_keypoint = affine_keypoint.reshape(-1)
    return affine_keypoint


def num_in_keyboard(all_points, matrix, kb_width, kb_height):
    count = 0
    for point in all_points:
        affine_keypoint = point_transform(point, matrix)
        count = count + 1 if is_in_keyboard(kb_width, kb_height, affine_keypoint) else count
    return count / len(all_points)


def coor_key_transform(keyboard_json, all_coors, matrix):
    ref_key_json_path = keyboard_json
    
    with open(ref_key_json_path) as f:
        ref_keyboard = (json.load(f))
    (x1_ref, y1_ref, x2_ref, y2_ref) = eval(ref_keyboard["keyboard"])    
    kb_width = x2_ref - x1_ref
    kb_height = y2_ref - y1_ref
    inside_ratio = num_in_keyboard(all_coors, matrix, kb_width, kb_height)

    for keypoint in all_coors:
        affine_keypoint = point_transform(keypoint, matrix)
        if not is_in_keyboard(kb_width, kb_height, affine_keypoint):
            if inside_ratio > 0.3:
                continue
            else:
                return None

        x_transformed = affine_keypoint[0] * 1 + x1_ref
        y_transformed = affine_keypoint[1] * 1 + y1_ref

        for key, value in ref_keyboard.items():
            if key == "keyboard":
                continue
            (x1, y1, x2, y2) = eval(value)
            if x_transformed > x1 and y_transformed > y1:
                if x_transformed < x2 and y_transformed < y2:
                    return key
        break
    
    return None  


