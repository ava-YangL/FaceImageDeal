import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2



predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


frame=cv2.imread('00001_A1.jpg')
dets = detector(frame, 0)
for k, d in enumerate(dets):
    shape = predictor(frame, d)
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    for num in range(shape.num_parts):
        cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
cv2.imwrite('now.png',frame)
#cv2.imshow('frame', frame)
'''
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        for num in range(shape.num_parts):
            cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break


cap.release()
out.release()

cv2.destroyAllWindows()

'''

