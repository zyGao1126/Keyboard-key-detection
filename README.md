# Keyboard key detection
Tracking the position of a finger on the keyboard and giving the value of the key.

## **Run process**
### **Step1: Reference keyboard registation**  
Firstly, we need to prepare a reference keybaord image and extract the boundary coordinates of all keys, then save them to the .json file. The relative code is under ./kb_detect folder. 

Execute:

`python main.py -mode=refKeyCalib`

<p align="center">
<img src="https://github.com/zyGao1126/Keyboard-key-detection/blob/master/sample/sample1.png" height="300px" style="margin-right: 10px;">
<img src="https://github.com/zyGao1126/Keyboard-key-detection/blob/master/sample/sample2.png" height="300px">
</p>

### **Step2: Actual environment registation**
Before actual test, we need to calibrate finger and keyboard based on the current scene and save parameters in .json files respectively.

#### Finger registation
Calibrate finger's color space.

`python main.py -mode=realFingCalib`

#### Keyboard registation
Calibrate keyboard's color space and boundary location.

`python main.py -mode=realKeyCalib`

### **Step3: Actual environment testing**  
After finishing the above calibration, we can begin our test. just execute:

`python main.py -mode=realTest`

Here is a simple demo:
<p align="center">
  <img src="https://github.com/zyGao1126/Keyboard-key-detection/blob/master/sample/sample_demo.gif" alt="demo" width="40%">
</p>
Note that, If the environment change (such as the keyboard being moved), please re-execuate step2-keyboard registation to update the keyboard location.

## **Acknowledgement**
This code is based on [KeyboardDetection](https://github.com/FlipGoncalves/KeyboardDetection) and [Finger-Detection-and-Tracking](https://github.com/amarlearning/Finger-Detection-and-Tracking).
