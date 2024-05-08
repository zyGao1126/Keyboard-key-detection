import os
import sys
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb_detect.img_calibration import get_binary_image
from kb_detect.vdo_calibration import extract_keyboard

# tmp
key_json_path = ""
vdo_json_path = ""

total_rectangle = 9
traverse_point = []
hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None

PURPLE = (255, 0, 255)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [7 * rows / 20, 7 * rows / 20, 7 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 11 * rows / 20,
         11 * rows / 20, 11 * rows / 20], dtype=np.uint32)
    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)
    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float64)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float64)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))
        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point, np.max(dist)
    return None, None

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    # thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)

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

def bias_calibration(far_point, cnt_centroid, far_dist, bias=15):
    dx = far_point[0] - cnt_centroid[0]
    dy = far_point[1] - cnt_centroid[1]
    
    bias_x = np.floor(bias * dx / far_dist) if dx > 0 else np.ceil(bias * dx / far_dist)
    bias_y = np.floor(bias * dy/ far_dist ) if dy > 0 else np.ceil(bias * dy / far_dist) 

    return tuple((np.uint32(far_point[0] - bias_x), np.uint32(far_point[1] - bias_y)))

def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    # cv2.imshow("hist_mask_image", rescale_frame(hist_mask_image))

    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image)
    if len(contour_list) < 1:
        return None
    max_cont = max(contour_list, key=cv2.contourArea)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, PURPLE, -1)

    hull = cv2.convexHull(max_cont, returnPoints=False)
    defects = cv2.convexityDefects(max_cont, hull)
    far_point, far_dist = farthest_point(defects, max_cont, cnt_centroid)
    if far_dist == None:
        return None
    calib_point = bias_calibration(far_point, cnt_centroid, far_dist)
    # print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point) + "calib point : " + str(calib_point))
    cv2.circle(frame, far_point, 5, RED, -1)
    cv2.circle(frame, calib_point, 5, WHITE, -1)    
    return calib_point

def coor_key_transform(opt, keypoint):
    real_key_json_path = opt.real_key_json_path
    ref_key_json_path = opt.ref_key_json_path
    
    with open(real_key_json_path) as f:
        real_keyboard = json.load(f)
    (x1_frame, y1_frame, x2_frame, y2_frame) = eval(real_keyboard["boundary"])    

    if keypoint[0] < x1_frame or keypoint[0] > x2_frame or keypoint[1] < y1_frame or keypoint[1] > y2_frame:
        print("The detection point is not in keyboard.")
    else:
        with open(ref_key_json_path) as f:
            ref_keyboard = (json.load(f))
        (x1_ref, y1_ref, x2_ref, y2_ref) = eval(ref_keyboard["keyboard"])
        
        scale_x = (x2_ref - x1_ref) / (x2_frame - x1_frame)
        scale_y = (y2_ref - y1_ref) / (y2_frame - y1_frame)
        x_transformed = (keypoint[0] - x1_frame) * scale_x + x1_ref
        y_transformed = (keypoint[1] - y1_frame) * scale_y + y1_ref
        print("scale_x: {}  scale_y: {}  x_transformed: {}  y_transformed: {}".format(scale_x, scale_y, x_transformed, y_transformed))
        for key, value in ref_keyboard.items():
            if key == "keyboard":
                continue
            (x1, y1, x2, y2) = eval(value)
            if x_transformed > x1 and y_transformed > y1:
                if x_transformed < x2 and y_transformed < y2:
                    print("***** Press {} *****".format(key))
                    break        

def real_finger_calib(opt):
    capture = cv2.VideoCapture(1)
    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        
        if pressed_key & 0xFF == ord('z'):
            print("** Finish finger histogram registeration ***")
            hand_hist = hand_histogram(frame)
            np.save(opt.finger_hist_path, hand_hist)
            break
        else:
            frame = draw_rect(frame)
        
        cv2.imshow("Finger histogram registeration", frame)
        if pressed_key == 27: #ESC
            break    
    cv2.destroyAllWindows()
    capture.release()      


