# Human Detection

This module is human detection module developed using OpenCV with SSD algorithm which will detect all the humans uniquely in real time. This module can also used to detect humans in video clip.

### Execution

* You can execute the code by typing "python human_detection.py" in your terminal and this will start the module and wont save anything on your system.
* If you want to run this module on a video clip and save the output on your system, use "python human_detection.py <INPUT_FILENAME> <OUTPUT_FILENAME>" in your terminal. For example "python human_detection.py test.mp4 output.avi" (Use python3 if using linux system).
* Speed can be varied by modifying line 107,note that accuracy also might vary.
* Other changes will be similar like the previous module.
* If the intruders count should be calculated and showed, then the 'record' variable should be set to true as soon as the employee punches the biometrics.
* The recording part and intruder part will be taken care and handled only when an employee punches the biometrics and sets the 'record' variable to true. The change in the value of this variable must be taken care by biometrics system.