import os
import cv2 
import math
import argparse
import numpy as np 
from process import *


def main():
    filemodel_1 = 'marker/joker.jpg'
    filemodel_2 = 'marker/xi.jpg'
    cap = cv2.VideoCapture(0)
    # Read frame and take bbox
    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 
        frame1 = detect_boundingbox(filemodel_1, frame, 120)
        frame2 = detect_boundingbox(filemodel_2, frame, 120)
        frame = cv2.addWeighted(frame1,0.5,frame2,0.5,0)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()
