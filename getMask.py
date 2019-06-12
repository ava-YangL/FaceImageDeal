# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:36:11 2017

@author: Quantum Liu
"""

import sys, os, traceback
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

class NoFace(Exception):
    '''
    没脸
    '''
    pass


class Organ():
    def __init__(self, im_bgr, im_hsv, temp_bgr, temp_hsv, landmark, name, ksize=None):
        '''
        五官部位类
        arguments:
            im_bgr:uint8 array, inference of BGR image
            im_hsv:uint8 array, inference of HSV image
            temp_bgr/hsv:global temp image
            landmark:array(x,2), landmarks
            name:string
        '''
        self.im_bgr, self.im_hsv, self.landmark, self.name = im_bgr, im_hsv, landmark, name
        self.get_rect()
        self.shape = (int(self.bottom - self.top), int(self.right - self.left))
        self.size = self.shape[0] * self.shape[1] * 3
        self.move = int(np.sqrt(self.size / 3) / 20)
        self.ksize = self.get_ksize()
        self.patch_bgr, self.patch_hsv = self.get_patch(self.im_bgr), self.get_patch(self.im_hsv)
        self.set_temp(temp_bgr, temp_hsv)
        self.patch_mask = self.get_mask_re()
        pass

    def set_temp(self, temp_bgr, temp_hsv):
        self.im_bgr_temp, self.im_hsv_temp = temp_bgr, temp_hsv
        self.patch_bgr_temp, self.patch_hsv_temp = self.get_patch(self.im_bgr_temp), self.get_patch(self.im_hsv_temp)

    def confirm(self):
        '''
        确认操作
        '''
        self.im_bgr[:], self.im_hsv[:] = self.im_bgr_temp[:], self.im_hsv_temp[:]

    def update_temp(self):
        '''
        更新临时图片
        '''
        self.im_bgr_temp[:], self.im_hsv_temp[:] = self.im_bgr[:], self.im_hsv[:]

    def get_ksize(self, rate=15):
        size = max([int(np.sqrt(self.size / 3) / rate), 1])
        size = (size if size % 2 == 1 else size + 1)
        return (size, size)

    def get_rect(self):
        '''
        获得定位方框
        '''
        ys, xs = self.landmark[:, 1], self.landmark[:, 0]
        self.top, self.bottom, self.left, self.right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)

    def get_patch(self, im):
        '''
        截取局部切片
        '''
        shape = im.shape
        return im[np.max([self.top - self.move, 0]):np.min([self.bottom + self.move, shape[0]]),
               np.max([self.left - self.move, 0]):np.min([self.right + self.move, shape[1]])]

    def _draw_convex_hull(self, im, points, color):
        '''
        勾画多凸边形
        '''
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_mask_re(self, ksize=None):
        '''
        获得局部相对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top - self.move, 0])
        landmark_re[:, 0] -= np.max([self.left - self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask,
                               landmark_re,
                               color=1)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        return cv2.GaussianBlur(mask, ksize, 0)[:]

    def get_mask_abs(self, ksize=None):
        '''
        获得全局绝对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        mask = np.zeros(self.im_bgr.shape, dtype=np.float64)
        print(self.im_bgr.shape)
        patch = self.get_patch(mask)
        patch[:] = self.patch_mask[:]
        return mask





class Face(Organ):
    '''
    脸类
    arguments:
        im_bgr:uint8 array, inference of BGR image
        im_hsv:uint8 array, inference of HSV image
        temp_bgr/hsv:global temp image
        landmarks:list, landmark groups
        index:int, index of face in the image
    '''

    def __init__(self, im_bgr, img_hsv, temp_bgr, temp_hsv, landmarks, index, path):
        self.index = index
        # 五官名称
        #landmarks = face_points(im_bgr)
        landmarks=getP(path) #CHANGE TO MY
        self.organs_name = ['left eye', 'right eye','all']
        '''
        
        for idx, point in enumerate(landmarks):
            pos = (point[0], point[1])
            # cv2.circle(im_bgr,pos,5,color=(0,255,0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im_bgr, str(idx), pos, font, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite("2.jpg", im_bgr)
                if num in[1, 12, 3, 59, 94, 34, 53, 67]:
            left_eye.append((dst_points[num][0], dst_points[num][1]))
        if num in [85, 104, 20, 27, 47, 43, 41, 51]:
                if (num < 20 or num >= 95 or num in [49, 56, 57, 66, 68, 76, 77, 78, 80, 81, 82]) and num not in [12, ,55] and num not in [91 ,85 ,104, 105, 27] and num not in [, 103]:
            face_points.append((dst_points[num][0], dst_points[num][1]))
        '''
        tempList=[]
        self.ill=[]

        for num in range(len(landmarks)):
            if (num < 20 or num >= 95 or num in [49, 56, 57, 66, 68, 76, 77, 78, 80, 81, 82]) and num not in [1, 12, 3,55] and num not in [91, 85, 104, 105, 20, 27] and num not in [19, 2, 4, 103, 42, 41, 115]:
            #if (num < 20 or num >= 95 or num in [49, 56, 57, 66, 68, 76, 77, 78, 80, 81, 82]) and num not in [1, 12, 3,55] and num not in [91, 85, 104, 105, 20, 27] and num not in [19, 2, 4, 103]:
                tempList.append(num)
                self.ill.append( (landmarks[num][0],landmarks[num][1]))

        #self.lll=tempList



        # 五官等标记点
        self.organs_points = [[1, 12, 3, 59, 94, 34, 53, 67],[85, 104, 20, 27, 47, 43, 41, 51],tempList]
        #self.organs_points = [list(range(30, 38)), list(range(40, 48))]
        # 实例化脸对象和五官对象
        self.organs = {name: Organ(im_bgr, img_hsv, temp_bgr, temp_hsv, landmarks[points], name) for name, points in
                       zip(self.organs_name, self.organs_points)}

        # 获得额头坐标，实例化额头
        self.mask_organs = self.organs['left eye'].get_mask_abs() + self.organs['right eye'].get_mask_abs() #+self.organs['all'].get_mask_abs()
        self.mask_facess=self.organs['all'].get_mask_abs()
        # 人脸的完整标记点

        self.FACE_POINTS = np.concatenate([landmarks])
        super(Face, self).__init__(im_bgr, img_hsv, temp_bgr, temp_hsv, self.FACE_POINTS, 'face')

        self.mask_face = self.get_mask_abs() - self.mask_organs
        self.mask_eye=self.mask_organs
        self.mask_all=self.mask_facess
        #self.mask_face = -self.mask_organs
        self.patch_mask = self.get_patch(self.mask_face)
        pass

    def get_forehead_landmark(self, im_bgr, face_landmark, mask_organs, mask_nose):
        '''
        计算额头坐标
        '''
        # 画椭圆
        radius = (np.linalg.norm(face_landmark[0] - face_landmark[16]) / 2).astype('int32')
        center_abs = tuple(((face_landmark[0] + face_landmark[16]) / 2).astype('int32'))

        angle = np.degrees(np.arctan((lambda l: l[1] / l[0])(face_landmark[16] - face_landmark[0]))).astype('int32')
        mask = np.zeros(mask_organs.shape[:2], dtype=np.float64)
        cv2.ellipse(mask, center_abs, (radius, radius), angle, 180, 360, 1, -1)
        # 剔除与五官重合部分
        mask[mask_organs[:, :, 0] > 0] = 0
        # 根据鼻子的肤色判断真正的额头面积
        index_bool = []
        for ch in range(3):
            mean, std = np.mean(im_bgr[:, :, ch][mask_nose[:, :, ch] > 0]), np.std(
                im_bgr[:, :, ch][mask_nose[:, :, ch] > 0])
            up, down = mean + 0.5 * std, mean - 0.5 * std
            index_bool.append((im_bgr[:, :, ch] < down) | (im_bgr[:, :, ch] > up))
        index_zero = ((mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        mask[index_zero] = 0
        index_abs = np.array(np.where(mask > 0)[::-1]).transpose()
        landmark = cv2.convexHull(index_abs).squeeze()
        return landmark


class Makeup():
    '''
    化妆器
    '''

    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.photo_path = []
        self.PREDICTOR_PATH = predictor_path
        self.faces = {}

        # 人脸定位、特征提取器，来自dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    def get_faces(self, im_bgr, im_hsv, temp_bgr, temp_hsv, name, n=1):
        '''
        人脸定位和特征提取，定位到两张及以上脸或者没有人脸将抛出异常
        im:
            照片的numpy数组
        fname:
            照片名字的字符串
        返回值:
            人脸特征(x,y)坐标的矩阵
        '''
        rects = self.detector(im_bgr, 1)

        if len(rects) < 1:
            raise NoFace('Too many faces in ' + name)
        return {name: [Face(im_bgr, im_hsv, temp_bgr, temp_hsv,
                            np.array([[p.x, p.y] for p in self.predictor(im_bgr, rect).parts()]), i, name) for i, rect
                       in enumerate(rects)]}

    def read_im(self, fname, scale=1):
        '''
        读取图片
        '''
        im = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
        if type(im) == type(None):
            print(fname)
            raise ValueError('Opencv error reading image "{}" , got None'.format(fname))
        return im

    def read_and_mark(self, fname):
        im_bgr = self.read_im(fname)
        im_bgr = cv2.merge((im_bgr, im_bgr, im_bgr))
        im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        temp_bgr, temp_hsv = im_bgr.copy(), im_hsv.copy()
        return im_bgr, temp_bgr, self.get_faces(im_bgr, im_hsv, temp_bgr, temp_hsv, fname)


import cv2
import numpy as np
import stasm


def boundary_points(points):
    """ Produce additional boundary points

    :param points: *m* x 2 np array of x,y points
    :returns: 2 additional points at the top corners
    """
    x, y, w, h = cv2.boundingRect(points)
    buffer_percent = 0.1
    spacerw = int(w * buffer_percent)
    spacerh = int(h * buffer_percent)
    return [[x + spacerw, y + spacerh],
            [x + w - spacerw, y + spacerh]]


def face_points(img, add_boundary_points=True):
    """ Locates 77 face points using stasm (http://www.milbo.users.sonic.net/stasm)

    :param img: an image array
    :param add_boundary_points: bool to add 2 additional points
    :returns: Array of x,y face points. Empty array if no face found
    """

    try:
        points = stasm.search_single(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    except Exception as e:
        print('Failed finding face points: ', e)
        return []

    points = points.astype(np.int32)
    if len(points) == 0:
        return points

    if add_boundary_points:
        return np.vstack([points, boundary_points(points)])

    return points


def average_points(point_set):
    """ Averages a set of face points from images

    :param point_set: *n* x *m* x 2 array of face points. \\
    *n* = number of images. *m* = number of face points per image
    """
    return np.mean(point_set, 0).astype(np.int32)


def weighted_average_points(start_points, end_points, percent=0.5):
    """ Weighted average of two sets of supplied points

    :param start_points: *m* x 2 array of start face points.
    :param end_points: *m* x 2 array of end face points.
    :param percent: [0, 1] percentage weight on start_points
    :returns: *m* x 2 array of weighted average points
    """
    if percent <= 0:
        return end_points
    elif percent >= 1:
        return start_points
    else:
        return np.asarray(start_points * percent + end_points * (1 - percent), np.int32)

def getP(path):
    dst_points = np.loadtxt(path[:-4] + '_106.txt').astype(int)
    # get head points
    sketch_frame_head = io.imread(path)
    img_s, dst_points_head = getPoint(sketch_frame_head)
    dst_points_head_choose = np.array(dst_points_head[68:81]).tolist()

    # extent two list
    dst_points = dst_points.tolist()
    dst_points.extend(dst_points_head_choose)
    dst_points = np.array(dst_points)
    return dst_points

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




plt.ion()
import os
if __name__ == '__main__':
    #initPath='/data/yl/code/Sketch/Data/CUFSF/getDataToTrainCycleGAN/2SketchSSizeFine/'
    #desPath='/data/yl/code/Sketch/Data/CUFSF/getDataToTrainCycleGAN/2SketchSSizeFine_Mask/'
    initPath='/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/Sketch/'
    desPath='/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/SketchMaskFinal2/'
    desPath1 = '/data/yl/code/Sketch/Data/CUFSF1/getSameLocationCUFSF/Res/temp/'
    imgPath=os.listdir(initPath)
    for i in range(5,len(imgPath)):
        path=initPath+imgPath[i]
        print(i)
        print(imgPath[i])
        mu = Makeup()
        im, temp_bgr, faces = mu.read_and_mark(path)
        imc = im.copy()



        for face in faces[path]:
            # img_temp = io.imread(path)
            # for num in range(len(face.ill)):
            #     cv2.circle(img_temp, (face.ill[num][0], face.ill[num][1]), 3, (0, 255, 0), -1)
            #
            # io.imsave(desPath + imgPath[i][:-4] + '_3des.jpg', img_temp)
            a = face.mask_face
            # plt.imshow(a)
            # a[abs(a - 1) > abs(a - 0)] = 0
            # a[abs(a - 0) > abs(a - 1)] = 1
            # c = a * im
            # cv2.imwrite(desPath+imgPath[i], a*255)
            # cv2.imwrite(desPath1 + imgPath[i], c)

            aaaa=face.mask_eye
            bbbb=face.mask_all

            kernel=np.ones((11,11),np.uint8)
            bbbb=cv2.erode(bbbb,kernel,iterations=1)

            plt.imshow(aaaa)
            plt.imshow(bbbb)
            a=bbbb-aaaa
            plt.imshow(a)
            a[abs(a - 1) > abs(a - 0)] = 0
            a[abs(a - 0) > abs(a - 1)] = 1
            a[a<1]=0
            c = a * im
            cv2.imwrite(desPath+imgPath[i], a*255)
            cv2.imwrite(desPath1 + imgPath[i], c)
            # for num in range(len(face.ill)):
            #     cv2.circle(a, (face.ill[num][0], face.ill[num][1]), 3, (0, 255, 0), -1)

            #io.imsave(desPath + imgPath[i][:-4] + '_4des.jpg', a)
            #cv2.imwrite(desPath + imgPath[i], a * 255)





