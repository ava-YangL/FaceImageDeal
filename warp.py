# -*- coding:utf-8 -*-
# @Time :4/25/19 6:02 PM
# @Author :Yangli
# @Site :
# @File :main.py
import numpy as np
from skimage import transform as tf
from skimage import io
from PIL import Image
import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2


def warp_image(image, src_points, dst_points):
    src_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + src_points  # .tolist()
    )
    dst_points = np.array(
        [
            [0, 0], [0, image.shape[0]],
            [image.shape[0], 0], list(image.shape[:2])
        ] + dst_points  # .tolist()
    )

    tform3 = tf.PiecewiseAffineTransform()
    tform3.estimate(dst_points, src_points)

    warped = tf.warp(image, tform3, output_shape=image.shape)
    return warped


def getPoint(frame):
    predictor_path = 'shape_predictor_81_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        for num in range(shape.num_parts):
            cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0, 255, 0), -1)
    return frame, landmarks


# photoPath='CUFSFPhotoFinal2/'
# cufsfPath='2SketchSSizeFine/'
# desPath='WarpDes/'
photoPath = '/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/Photo/'
cufsfPath = '/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/Sketch/'
desPath = '/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/Warp1/'

photos = os.listdir(photoPath)
photos = [x for x in photos if 'A.jpg' in x]
sketchs = os.listdir(cufsfPath)
sketchs = [x for x in sketchs if 'jpg' in x]
print(len(photos))
print(len(sketchs))

for i in range(len(photos)):
    photoname = photoPath + photos[i]
    sketchname = cufsfPath + sketchs[i]
    desname = desPath + photos[i]

    img_frame = io.imread(photoname)
    img_init = io.imread(photoname)
    sketch_frame = io.imread(sketchname)
    sketch_init = io.imread(sketchname)

    # get face++ points
    src_points = np.loadtxt(photoname[:-4] + '_106.txt').astype(int)
    dst_points = np.loadtxt(sketchname[:-4] + '_106.txt').astype(int)

    # get head points
    img_frame_head = io.imread(photoname)
    sketch_frame_head = io.imread(sketchname)
    img_p, src_points_head = getPoint(img_frame_head)
    img_s, dst_points_head = getPoint(sketch_frame_head)
    src_points_head_choose = np.array(src_points_head[68:81]).tolist()
    dst_points_head_choose = np.array(dst_points_head[68:81]).tolist()

    # extent two list
    src_points = src_points.tolist()
    dst_points = dst_points.tolist()
    src_points.extend(src_points_head_choose)
    dst_points.extend(dst_points_head_choose)

    img_res = warp_image(img_init, src_points, dst_points)
    if (len(src_points) != len(dst_points)):
        print(photoname)

    io.imsave(desname[:-4] + '.jpg', img_res)

    for num in range(len(dst_points)):
        # print(dst_points[num][0])
        cv2.circle(img_frame, (src_points[num][0], src_points[num][1]), 3, (0, 255, 0), -1)
    for num in range(len(dst_points)):
        # print(dst_points[num][0])
        cv2.circle(sketch_frame, (dst_points[num][0], dst_points[num][1]), 3, (0, 255, 0), -1)
    io.imsave(desname[:-4] + '_0sketch.jpg', sketch_init)
    io.imsave(desname[:-4] + '_1img.jpg', img_frame)
    io.imsave(desname[:-4] + '_2sketch.jpg', sketch_frame)
    for num in range(len(dst_points)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img_res, str(num), (dst_points[num][0], dst_points[num][1]), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.circle(img_res, (dst_points[num][0], dst_points[num][1]), 1, (0, 255, 0), -1)
    #io.imsave(desname[:-4] + '_3des.jpg', img_res)

    # dst_points = dst_points.tolist()
    face_points = []
    left_eye = []
    right_eye = []

    for num in range(len(dst_points)):
        if (num < 20 or num >= 95 or num in [49, 56, 57, 66, 68, 76, 77, 78, 80, 81, 82]) and num not in [1, 12, 3,55] and num not in [91 ,85 ,104, 105, 20, 27] and num not in [19, 2, 4, 103,42,41,115]:
            cv2.putText(img_res, str(num), (dst_points[num][0], dst_points[num][1]), font, 0.5, (0, 0, 255), 1,cv2.LINE_AA)
            cv2.circle(img_res, (dst_points[num][0], dst_points[num][1]), 3, (0, 255, 0), -1)
            face_points.append((dst_points[num][0], dst_points[num][1]))
        if num in[1, 12, 3, 59, 94, 34, 53, 67]:
            left_eye.append((dst_points[num][0], dst_points[num][1]))
        if num in [85, 104, 20, 27, 47, 43, 41, 51]:
            right_eye.append((dst_points[num][0], dst_points[num][1]))

    for num in range(len(face_points)):
        a=1
        #cv2.circle(img_res, (face_points[num][0], face_points[num][1]), 3, (0, 255, 0), -1)
        #cv2.putText(img_res, str(num), (face_points[num][0], face_points[num][1]), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    for num in range(len(left_eye)):
        cv2.circle(img_res, (left_eye[num][0], left_eye[num][1]), 3, (0, 255, 0), -1)
    for num in range(len(right_eye)):
        cv2.circle(img_res, (right_eye[num][0], right_eye[num][1]), 3, (0, 255, 0), -1)
    io.imsave(desname[:-4] + '_3des.jpg', img_res)

    a=1;

'''
frame = io.imread('00003_A.jpg')
img_init=io.imread('00003_A.jpg')
img_p,src_points=getPoint(frame)

frame1 = io.imread('00003_A1.jpg')
img_s,dst_points=getPoint(frame1)

img_res=warp_image(img_init, src_points, dst_points)
io.imsave("3.png",img_res)
dst_points=dst_points.tolist()
for num in range(len(dst_points)):
    print(dst_points[num][0])
    cv2.circle(img_res, (dst_points[num][0], dst_points[num][1]), 3, (0, 255, 0), -1)
cv2.imwrite("1.png",img_p)
cv2.imwrite("2.png",img_s)
io.imsave("4.png",img_res)
'''
