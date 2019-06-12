# -*- coding:utf-8 -*-
# @Time :4/26/19 5:02 PM
# @Author :Yangli
# @Site :
# @File :face++.py
# 通过face++的接口获得106个关键点，不包括额头


# -*- coding: utf-8 -*-
# import json
import math
import os
import threading
import time
import urllib.request
import urllib.error
import json as js
import numpy

KEYS = []
SECRETS = []
'''
KEYS和SECRETS是face++的接口获得的，这里隐匿掉了
'''


FAILED_IMAGES = []
HTTP_URL = "https://api-cn.faceplusplus.com/facepp/v3/face/analyze"

 DATA_ROOT='Sketch/'

TARGET_ROOT = ''

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
                  '.bmp', '.BMP']

RESULTS = []


class MyThread(threading.Thread):
    def __init__(self, image_list, thread_id):
        threading.Thread.__init__(self)
        self.image_list = image_list
        self.thread_id = thread_id

    def run(self):
        print('thread {} begin'.format(self.thread_id))
        post_to_facepp(self.image_list, self.thread_id)
        print('thread {} end.'.format(self.thread_id))


def list_all_images(dir_root):
    image_list = []
    a=[]
    b=[]
    imgs=os.listdir(dir_root)
    for i in range(len(imgs)):
        if('txt'in imgs[i]):
            a.append(imgs[i])
        if('_A.jpg' in imgs[i]):
            b.append(imgs[i])
    a=set(a)
    for i in range(len(b)):
        #if(b[i][:-4]+'106.txt' not in a):
            image_list.append(dir_root+b[i])
    print(len(image_list))
    return image_list

def post_data(file_path, thread_id):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(KEYS[thread_id])
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(SECRETS[thread_id])
    data.append('--%s' % boundary)
    fr = open(file_path, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('2')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append('none')
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url="https://api-cn.faceplusplus.com/facepp/v3/detect", data=http_body)
    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

                    
    return req


def new_name(image_path):
    dirname, filename = os.path.split(image_path)
    new_dirpath = str(dirname).replace(DATA_ROOT, TARGET_ROOT)
    if not os.path.exists(new_dirpath):
        try:
            os.makedirs(new_dirpath)
        except:
            pass

    index = filename.rfind('.')
    json_name = os.path.join(new_dirpath, filename[:index] + '.json')

    return json_name


def save_to_disk(json_name, json_str):
    with open(json_name, 'w') as f:
        json.dump(json_str, f)


def post_to_facepp(image_list, thread_id):
    image_len = len(image_list)
    print_interval = image_len // 100
    print_interval = print_interval if print_interval > 0 else 1
    #print(image_list)
    for index, image_path in enumerate(image_list):
        req = post_data(image_path, thread_id)
        try:
            #print("22222222222222222222222222222222222222")
            resp = urllib.request.urlopen(req, timeout=5)
            # get response
            qrcont = resp.read()
            #print(qrcont)
             
            mydict = eval(qrcont.decode('utf-8'))
            faces=mydict["faces"] 
            faces=faces[0]
            #print(faces)
            #time.sleep(100)
            landmark=faces['landmark']

            #print(landmark)
            #time.sleep(100)

            subkeys=list(landmark.keys())
            points=[]
            #print(subkeys)
            #time.sleep(100)
            for j in range(len(subkeys)):

                #now=landmark[initkeys[i]]
                #subkeys=list(now.keys())
                # print(subkeys)
                #print('%%%%%%%%%%%')
                #for j in range(len(subkeys)):
                #print("$$$$$$$$$$$$$$$")
                #print(subkeys[j])
                
                if(subkeys[j]!='face_token'):
                    if(subkeys[j]!='face_rectangle'):
                        #print(subkeys[j],landmark[subkeys[j]])#['x'],landmark[subkeys[j]]['y'])
                        #time.sleep(10)
                        temp=[landmark[subkeys[j]]['x'],landmark[subkeys[j]]['y']]
                        #print(temp)
                        points.append(temp)
                        
            #print(points)
            print(image_path)
            numpy.savetxt(image_path[:-4]+'_106.txt',np.array(points))
            if index % print_interval == 0 and index > 0:
                print('{}/{} in thread {} has been sloven'
                      .format(index, image_len, thread_id))
            #print(point)

            #numpy.savetxt(image_path[:-4]+'.txt',np.array(pointP))
            #print("$$")

        except:# urllib.error.HTTPError as e:
            #print("22222")
            #print(e.read().decode('utf-8'))
            FAILED_IMAGES.append(image_path)


def start_threads(image_list, n_threads=4):
    if n_threads > len(image_list):
        n_threads = len(image_list)
    n = int(math.ceil(len(image_list) / float(n_threads)))
    print('the thread num is {}'.format(n_threads))
    print('each thread images num is {}'.format(n))
    image_lists = [image_list[index:index + n] for index in range(0, len(image_list), n)]
    thread_list = {}
    for thread_id in range(n_threads):
        thread_list[thread_id] = MyThread(image_lists[thread_id], thread_id)
        thread_list[thread_id].start()

    for thread_id in range(n_threads):
        thread_list[thread_id].join()





import numpy
import numpy as np

#a=np.loadtxt("1.txt")
import time

if __name__ == '__main__':
    DATA_ROOT='Sketch/'
    image_list = list_all_images(DATA_ROOT)
    #print(image_list)
    print('all images num is {}'.format(len(image_list)))
    
    start_threads(image_list, len(KEYS))
    last_failed_num = 0
    for i in range(100):
        this_failed_num = len(FAILED_IMAGES)
        print('before {}th resolve failed images, the failed images num is {}.'
              .format(i + 1, this_failed_num))

        if last_failed_num == this_failed_num:
            break
        else:
            last_failed_num = this_failed_num

        new_image_list = [image_path for image_path in FAILED_IMAGES]
        FAILED_IMAGES.clear()
        start_threads(new_image_list, len(KEYS))

    print('the num of the image failed is {}'.format(len(FAILED_IMAGES)))
    print('the num of the result you get is {}'.format(len(RESULTS)))

    with open(os.path.join(TARGET_ROOT, 'FAILED_IMAGES.txt'), 'w', encoding='utf-8') as f:
        for image_path in FAILED_IMAGES:
            try:
                f.write('{}\n'.format(image_path))
            except:
                print(image_path)
