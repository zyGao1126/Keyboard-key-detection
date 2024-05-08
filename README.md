# Keyboard key detection
Tracking the position of a finger on the keyboard and giving the value of the key.

## **Run process**
### **Step1: Reference keyboard registation**  
Firstly, we need to prepare a reference keybaord image and extract the boundary coordinates of all keys, then save them to the .json file. The relative code is under ./kb_detect folder. 

Execute:

`python kb_dect/img_calibration.py`

### **Step2: Actual environment registation**
Before actual test, we need to calibrate finger and keyboard based on the current scene and save parameters in .json files respectively.

#### Finger registation
Update finger's color space.

#### Keyboard registation
Update keyboard's color space and boundary location.

`python kb_dect/vdo_calibration.py`


### **Step3: Actual environment testing**  
After finishing the above registation, we can begin our test. just execute:

`python main.py`

Note that, If the environment change (such as the keyboard being moved), please re-execuate step2-keyboard registation to update the keyboard location.

## **Acknowledgement**
This code is based on [KeyboardDetection](https://github.com/FlipGoncalves/KeyboardDetection) and [Finger-Detection-and-Tracking](https://github.com/amarlearning/Finger-Detection-and-Tracking).