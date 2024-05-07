# Keyboard key detection
Tracking the position of a finger on the keyboard and giving the value of the key.

## **Run process**
### **Step1: Reference keyboard registation**  
Firstly, we need to prepare a reference keybaord image and extract the boundary coordinates of all keys, then save them to the .json file. The relative code is under ./kb_detect folder. 

Execute:  
`python kb_dect/img_calibration.py`

### **Step2: Actual env keyboard registation**   
Before actual test, we need to calibrate the keyboard image in the test environment first. The keyboard's color space will be updated based on the current scene and saved in another .json file.

Execute:  
`python kb_dect/vdo_calibration.py`

### **Step3: Actual env testing**  
Then we can start the real test.

Execute:  
`python main.py`

