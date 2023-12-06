import os
import re
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Functions:
    parse__pic_name
    parse__gt_name
    path_join
    read_names
    build_dataset
    show_data_sample
    resize
"""


"""
Classes:
    DataGenerator
"""





def resize(img, target=256, is_3d=True):
    h, w = img.shape[:2]
    if w > h:
        ratio = h / w
        new_w = int(target * ratio)
        re_size = (target, new_w)
        upper = int((target - new_w) / 2)
        lower = target - new_w - upper
        paddings = [upper, lower, 0, 0]
    else:
        ratio = w / h
        new_h = int(target * ratio)
        re_size = (new_h, target)
        left = int((target - new_h) / 2)
        right = target - new_h - left
        paddings = [0, 0 , left, right]
    if is_3d:
        # w is target
        img_resize = cv2.resize(img, re_size)
    else:
        img_resize = cv2.resize(img, re_size, interpolation=cv2.INTER_NEAREST)
    # padding cv.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=BORDER_REFLECT_101)
    try:
        img_padding = cv2.copyMakeBorder(img_resize,paddings[0],paddings[1],paddings[2],
                                        paddings[3],borderType=cv2.BORDER_CONSTANT, value=0)
    except Exception:
        print(paddings, new_w, lower, upper, img.shape)
    return img_padding


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

def build_dataset(pic_names, gt_names):
    '''Return a dict where pic: is the train_pic'''
    datas = dict()
    # build the keys
    for pic_name in pic_names:
        id = parse_pic_name(pic_name)
        datas[id] = dict()
        datas[id]['x'] = pic_name
    # build the values
    for gt_name in gt_names:
        id = parse_gt_name(gt_name)
        datas[id]['y'] = gt_name
    # check the dict
    for _, v in datas.items():
        assert len(v.items()) == 2
    datas_list = list(datas.values())
    return datas_list

def show_data_sample(x, y):
    print('bshow')
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].imshow(x)
    ax[1].imshow(y)
    plt.savefig('t.png')
    plt.close()
    
class DataGenerator(object):
    def __init__(self, datas) -> None:
        self._datas = datas
        self._dlen = len(datas)
    
    def data_iter(self, batch_size = 128, random=True):
        
        def extract_xs_ys(idxs):
            xs = list()
            ys = list()
            for idx in idxs:
                d = self._datas[idx]
                assert parse_pic_name(d['x']) == parse_gt_name(d['y'])
                # read the data
                pic = cv2.imread(d['x'], cv2.IMREAD_UNCHANGED)
                gt = cv2.imread(d['y'], cv2.IMREAD_UNCHANGED)
                gt[gt > 0] = 1
                assert pic.shape[:2] == gt.shape[:2]
                xs.append(resize(pic, target=128))
                ys.append(resize(gt, target=128, is_3d=False))
            return np.array(xs), np.array(ys)
        
        if random:
            idxs = np.random.permutation(self._dlen)
        else:
            idxs = list(range(self._dlen))
        
        batchs = self._dlen // batch_size
        rest = self._dlen % batch_size
        batchs_ = batchs + 1 if rest else batchs
        for bi in range(batchs_):
            if bi == batchs:
                bs = rest
            else:
                bs = batch_size
            names = list(self._datas[i] for i in idxs[bi*batch_size: bi*batch_size + bs])
            yield extract_xs_ys(idxs[bi*batch_size: bi*batch_size + bs]), bi, names


def get_train_generator(tg, batch_size=128, random=True):
    return tg.data_iter(batch_size=batch_size, random=random)


# if __name__ == "__main__":
#     # # train_names = read_file_names(TRAIN_PATH)
#     # # train_gt_names = read_file_names(TRAIN_GT_PATH)
#     # print('ds f')
#     # print('ds f2')
#     # ti = tg.data_iter(batch_size=4)
#     # print('ds f3')
#     # for xs, ys in ti:
#     #     print(xs.shape, ys.shape)
    
#     # xt = r"D:\dl\datas\skin18\train_data\ISIC_0000000.jpg"
#     # xgt = r"D:\dl\datas\skin18\train_gt\ISIC_0000000_segmentation.png"
    
#     # p = cv2.imread(xt, cv2.IMREAD_UNCHANGED)
#     # gt = cv2.imread(xgt, cv2.IMREAD_UNCHANGED)
#     # p = resize(p)
#     # gt = resize(gt, is_3d=False)
#     # show_data_sample(p, gt)