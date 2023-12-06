import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import threading
import time
import re
from multiprocessing import Process, Manager
import queue
from multiprocessing import Process, Queue
import shutil

sep = os.sep

def mkdir(path):
    """
    Make sure the path is empty
    """
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
        print(f"Making path: {path} success")
    else:
        shutil.rmtree(path)
        os.mkdir(path)
        print(f"Delete and Making path: {path} success")



class NetSaver(object):
    
    def __init__(self, save_points, save_nums=1, tolerance=0.001):
        self._save_points = save_points
        self._ckpts = [save_nums] * len(self._save_points)
        self._toler = tolerance
        
    def check_value(self, value):
        max_i = 0
        max_v = 0
        for i in range(len(self._save_points)):
            if value > (self._save_points[i] - self._toler) and self._save_points[i] > max_v:
                max_v = self._save_points[i]
                max_i = i
        if self._ckpts[max_i] > 0:
            self._ckpts[max_i] -= 1
            return True
        else:
            return False       



def parse_pic_name(fname) -> int:
    'get the pic id from the pic path name'
    pat = '\_[0-9]+\.'
    id = re.findall(pat, fname)
    assert len(id) == 1, (id, fname)
    return int(re.findall('\d+', id[0])[0])

def parse_gt_name(fname) -> int:
    'get the gt id from the pic path name'
    pat = '\_[0-9]+\_'
    id = re.findall(pat, fname)
    assert len(id) == 1, (id, fname)
    return int(re.findall('\d+', id[0])[0])

lock = threading.Lock()


def resize_padding(img, target=256, is_3d=True):
    pass

def add_uniform_noise(image, prob=0.05, value=255):
    pass
    # return output



def aug_V1(img, gt):
    # assume my pic has not been resize
    # set the probs
    crop_prob   = 0.362
    is_crop     = 0
    flip_prob   = 0.5
    rotate_prob = 0.5
    # noise_prob  = 0.362
    noise_prob  = 0.362
    gauss_prob  = 0.362
    # get the h w
    h, w = img.shape[:2]
    # get the copy
    _img = img.copy()
    _gt  = gt.copy()
    # crop
    crop = np.random.random()
    if crop > crop_prob:
        # crop and resize cut by a point of the left up
        # caculate the size size is random but, min scale is 50%
        pass
    # flip
    flip = np.random.random()
    if flip > flip_prob:
        pass
        
    # rotate
    rotate = np.random.random()
    if rotate > rotate_prob:
        pass
    # gauss
    gauss = np.random.random()
    if gauss > gauss_prob:
        pass
    # noise
    noise = np.random.random()
    if noise > noise_prob:
        pass
    # resize and padding
    if is_crop:
        pass
    else:
        pass
    return _img, _gt






def read_img(save_path, datas, idx, is_train):
    x_path = f'{save_path}{sep}{datas[idx]}{sep}x.jpg'
    y_path = f'{save_path}{sep}{datas[idx]}{sep}y.png'
    x = cv2.imread(x_path, cv2.IMREAD_UNCHANGED)
    y = cv2.imread(y_path, cv2.IMREAD_UNCHANGED)
    if is_train:
        x, y = aug_V1(x, y)
    else:
        x = resize_padding(x)
        y = resize_padding(y, is_3d=False)
    return x, y

class MyReaderThread(threading.Thread):

    pass


# def MuAugPro()SSSS

class DataGenerator(object):
    def __init__(self, save_path, is_train=True, nums=None) -> None:
        self._save_path = save_path
        self._datas = os.listdir(save_path)
        if nums: 
            self._datas = os.listdir(save_path)[:nums]
        self._dlen = len(self._datas)
        self._is_train = is_train
    
    
    def resize_padding(self, img, target=256, is_3d=True):
        pass
    
    def add_uniform_noise(self, image, prob=0.05, value=255):
        pass
    
    def aug_V1(self, img, gt):
        # assume my pic has not been resize
        # set the probs
      pass

    def multiple_thread_read_img(self, idxs, num_threads_=16):
        if len(idxs) > 16:
            num_threads = 16
        else:
            num_threads = 1
        xs = list()
        ys = list()
        ids = list()
        threads = list()
        start = time.time()
        thread_tasks = len(idxs) // num_threads
        rest = len(idxs) % num_threads
        for i in range(num_threads):
            if i == num_threads - 1:
                _idxs = idxs[i*thread_tasks: (i+1)*thread_tasks+rest]
            else:
                _idxs = idxs[i*thread_tasks: (i+1)*thread_tasks]
                
            t = MyReaderThread(save_path=self._save_path,
                               idxs=_idxs, 
                               xs=xs, 
                               ys=ys,
                               ids=ids, 
                               id=i,
                               is_train=self._is_train,
                               aug=self.aug_V1)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print(f'Batch read finish, using :{time.time() - start} s')
        xs = np.array(xs)
        ys = np.array(ys)
        
        return xs, ys, ids
    
    def get_DataIter(self, batch_size=32):
        pass


def read_img_fn(xs, ys, ids, q, ist):
    for i, (x, y, id) in enumerate(zip(xs, ys, ids)):
        if ist:
            x, y = aug_V1(x, y)
        else:
            x = resize_padding(x)
            y = resize_padding(y, is_3d=False)
        # print('finish==========',i)
        q.put([x, y, id])
        # print(q.qsize(), "+++++++++2")
    # print('begin', len(xs), len(ys)
    return 0
        


if __name__ == "__main__":
    mkdir('train')
    mkdir('train/dice')
    mkdir('train/loss')
    mkdir('train/ce')
    if int(sys.argv[1]) == 0:

        batch_size = 64
        num_epochs = 10
        TRAIN_SAVE_PATH = r"D:\dl\datas\skin18\train"
        TEST_SAVE_PATH = r"D:\dl\datas\skin18\test"
            
    elif int(sys.argv[1]) == 1:
        TRAIN_SAVE_PATH = r"/home/miaomukang/datasets/isic18/train"
        TEST_SAVE_PATH = r"/home/miaomukang/datasets/isic18/test"
        batch_size = 64
        num_epochs = 150


    D = DataGenerator(TRAIN_SAVE_PATH, is_train=False)
    di = D.get_DataIter(batch_size=batch_size)
    for xs, ys, bi, ns in di:
        print(ns[0], xs.shape)
        print('save', f'pics{sep}{ns[0]}x.jpg')
        plt.imsave(f'pics{sep}{ns[0]}x.jpg', xs[0])
        plt.imsave(f'pics{sep}{ns[0]}y.jpg', ys[0])
        print('save2')
        
        # break