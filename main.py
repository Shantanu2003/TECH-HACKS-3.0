import os

import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier('D:/SHANTANU/TECH HACKS 3.0/keras_model.h5', 'D:/SHANTANU/TECH HACKS 3.0/labels.txt')

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    predection = classifier.getPrediction(img)
    print(predection)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
