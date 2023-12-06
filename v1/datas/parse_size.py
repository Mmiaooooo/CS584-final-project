import os
import re
import cv2
import time
import threading

"""
Functions:
    path_join
    read_names
"""




def path_join(fp1, fp2) -> str:
    """Return the join path"""
    return os.path.join(fp1, fp2)

def read_file_names(path) -> list:
    names = list()
    for f in os.listdir(path):
        ftype = (re.findall('\.[a-zA-Z]+', f))[-1]
        if ftype == '.png' or ftype == '.jpg':
            names.append(path_join(path, f))
    return names

l = list()
lock = threading.Lock()             # 实例化一个锁

class MyReaderThread(threading.Thread):

    def __init__(self, fns, ls, id):
        self._pns = fns
        self._ls = ls
        self._id = id
        self._ds = list()
        super().__init__()

    def run(self) -> None:
        global l
        for fn in self._pns:
            p = cv2.imread(fn, cv2.IMREAD_UNCHANGED).shape[:2]
            h, w = p
            if h > w:
                print(fn, p)
                
            self._ds.append(p)
        with lock:
            self._ls.extend(self._ds)



TRAIN_PATH = r"D:\dl\datas\skin18\train_data"
TRAIN_GT_PATH = r"D:\dl\datas\skin18\train_gt"

train_names = read_file_names(TRAIN_PATH)
train_gt_names = read_file_names(TRAIN_GT_PATH)

pic_sizes = list()
ts = list()
bts = len(train_names) // 10
for i in range(10):
    t = MyReaderThread(train_names[i*bts: (i+1)*bts], pic_sizes, i)
    ts.append(t)
    t.start()
for t in ts:
    t.join()


gt_sizes = list()
ts = list()
bts = len(train_names) // 10
for i in range(10):
    t = MyReaderThread(train_gt_names[i*bts: (i+1)*bts], gt_sizes, i)
    ts.append(t)
    t.start()
for t in ts:
    t.join()
    
print(set(pic_sizes) - set(gt_sizes))
print(set(pic_sizes))
print(set(gt_sizes))
