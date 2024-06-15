import yaml
import queue
import threading
import cv2
import json
import numpy as np
from finger_detect.finger_calibration import manage_image_opr, coor_key_transform
from kb_detect.ini_calibration import processImage
from kb_detect.real_calibration import get_contour
from finger_detect.real_test import matrixChange, isLegalKeyboard

class KBFrameProcessor:

    instance = None

    @classmethod
    def getInstance(cls):
        if cls.instance is None:
            cls.instance = KBFrameProcessor()
        return cls.instance

    def __init__(self, config_path="./config.yaml"):
        if KBFrameProcessor.instance is not None:
            raise Exception("KBFrameProcessor is a singleleton!")
        self.q = queue.Queue()
        self.q_result = queue.Queue()
        # for auto-detect keyboard location
        self.ref_area = None
        self.last_area = None
        self.ref_matrix = None
        self.last_matrix = None
        self.count = 0
        # for config parameters
        self.cfg_path = config_path
        self.cfg_refKey, self.cfg_finger, self.cfg_realKey, self.cfg_test = self.load_config(self.cfg_path)  
        # launch a thread to start detection
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def load_config(self, cfg_path):
        with open(cfg_path, 'r') as file:
            config = yaml.safe_load(file)
        cfg_refKey = config.get('refKeyCalib')
        cfg_finger = config.get('realFingerCalib')
        cfg_realKey = config.get('realKeyCalib')
        cfg_test = config.get('realTest')
        return cfg_refKey, cfg_finger, cfg_realKey, cfg_test     

    def fifo_update(self, area, matrix):
        if area is not None:
            if self.ref_area is None:
                if self.count < 5:
                    self.count = self.count + 1 if (not matrixChange(self.cfg_test, self.last_matrix, matrix) and isLegalKeyboard(area)) else 0
                else:
                    self.ref_area = self.last_area 
                    self.ref_matrix = self.last_matrix
                self.last_area = area
                self.last_matrix = matrix 
            else:
                area_th = self.ref_area * 0.05
                if abs(area - self.ref_area) < area_th and matrixChange(self.cfg_test, self.ref_matrix, matrix):
                    self.ref_area = self.ref_matrix = None
                    self.last_area = area
                    self.last_matrix = matrix 
                    self.count = 0    

    def run_kbDetect(self, frame, limit_real="./json_file/limit_real.json"):
        with open(limit_real) as f:
            ranges = (json.load(f))["limits"]   
        binary_image = cv2.cvtColor(processImage(ranges, frame), cv2.COLOR_BGR2GRAY)
        _, _, matrix, area = get_contour(self.cfg_realKey, binary_image, frame, self.cfg_refKey.get('ref_key_json_path'))          
        self.fifo_update(area, matrix)
        hand_hist = np.load(self.cfg_finger.get('finger_hist_path'))
        coor = manage_image_opr(frame, hand_hist, self.cfg_test.get('coor_bias'))
        key = None
        if coor and self.ref_matrix is not None:
            key = coor_key_transform(self.cfg_refKey.get('ref_key_json_path'), coor, self.ref_matrix)
        # cv2.imshow("Keyboard operator detection", frame)
        return key

    def _reader(self):
        while True:
            if not self.q.empty():
                frame = self.q.get(block=True, timeout=None)
                result = self.run_kbDetect(frame)
                self.q_result.put(result)

    def put(self, frame):
      self.q.put(frame)

    def get(self):
        result = self.q_result.get()
        return result

def main():
    KBprocess = KBFrameProcessor.getInstance()

    capture = cv2.VideoCapture(1) 
    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        
        KBprocess.put(frame)

        if pressed_key == 27: #ESC
            break         
    cv2.destroyAllWindows()
    capture.release()        


main()    

