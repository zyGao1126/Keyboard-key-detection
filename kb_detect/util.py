import numpy as np
import cv2
import copy
import json

def write_json(filename, data):
    with open(filename, 'w') as file_handle:
        json.dump({"limits": data}, file_handle)