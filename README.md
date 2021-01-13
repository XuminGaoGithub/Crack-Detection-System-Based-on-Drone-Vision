#Crack Detection System Based on Drone Vision

#Authur：Xumin Gao, Bin Lei (Institute of Robotics and Intelligent Systems, Wuhan University of Science and Technology,China)


#E-mail: comin15071460998@gmail.com

#Demo: https://www.bilibili.com/video/BV1na4y1E74n/

# Requirements

- Opencv 2.4.9




# Introduction

###Report/

-The detailed summary

###train/

-The training dataset

###test/

-The test dataset



###arduino_Ultrasonic.ino
-We use Arduino to get the distance between the camera and the crack.

###SerialPort.cpp and SerialPort.h

-The serial port which is used to access the Ultrasonic data from Arduino.


###fenleiqi.cpp and fenleiqi.h

-The algorithm of crack segmentation and detection 



#Main function

###1. main_train.cpp

-The file to train the model which is used to classify different types of cracks.


###2. main_test_camera&ultrasonic.cpp

-Using camera to conduct the detection of cracks. Using ultrasonic data and triangle rule to calculate the width and area of cracks in the actual environment

###SVM_DATA.xml

-The trained model


#Abstract

We use image processing method (the adaptive threshold segmentation) to segment the cracks. At the same time, we extract some significant features (area, the horizontal and vertical integral projection, distribution density   of crack in images) to train the classifier, which can recognize different types of cracks. Finally, we use ultrasonic data and the triangle rule to calculate the width and area of cracks in the actual environment.