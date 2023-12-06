import cv2
import matplotlib.pyplot as plt
import re
import os
import shutil
import sys



def resize2Target(img, target=256, is_3d=True):
    pass


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
        datas[id]['id']=id
        datas[id]['x'] = pic_name
    # build the values
    for gt_name in gt_names:
        id = parse_gt_name(gt_name)
        datas[id]['y'] = gt_name
    # check the dict
    for _, v in datas.items():
        assert len(v.items()) == 3
    datas_list = list(datas.values())
    return datas_list

def mkdir(path):
    pass

def split_resize_save_img(dataset, save_path):
    mkdir(save_path)
    for data in dataset:
        
        img_path = path_join(save_path, str(data['id']))
        mkdir(img_path)
        x_path = data['x']
        y_path = data['y']
        xf_name = path_join(img_path, 'x.jpg')
        yf_name = path_join(img_path, 'y.png')
        
        # x = resize_and_padding(cv2.imread(x_path, cv2.IMREAD_UNCHANGED))
        # y = resize_and_padding(cv2.imread(y_path, cv2.IMREAD_UNCHANGED), is_3d=False)
        
        x = cv2.imread(x_path, cv2.IMREAD_UNCHANGED)
        y = cv2.imread(y_path, cv2.IMREAD_UNCHANGED)
        x = resize2Target(x, target=512, is_3d=True)
        y = resize2Target(y, target=512, is_3d=False)
        cv2.imwrite(xf_name, x)
        cv2.imwrite(yf_name, y)




if int(sys.argv[1]) == 0:

    batch_size = 64
    num_epochs = 10
    TRAIN_PATH = r"D:\dl\datas\skin18\train_data"
    TRAIN_GT_PATH = r"D:\dl\datas\skin18\train_gt"
    TRAIN_SAVE_PATH = r"D:\dl\datas\skin18\train"
    
    TEST_PATH = r"D:\dl\datas\skin18\test_data"
    TEST_GT_PATH = r"D:\dl\datas\skin18\test_gt"
    TEST_SAVE_PATH = r"D:\dl\datas\skin18\test"
        
elif int(sys.argv[1]) == 1:
    TRAIN_PATH = r"/home/miaomukang/datasets/isic18/train_data"
    TRAIN_GT_PATH = r"/home/miaomukang/datasets/isic18/train_gt"
    TRAIN_SAVE_PATH = r"/home/miaomukang/datasets/isic18/train"
    
    TEST_PATH = r"/home/miaomukang/datasets/isic18/test_data"
    TEST_GT_PATH = r"/home/miaomukang/datasets/isic18/test_gt"
    TEST_SAVE_PATH = r"/home/miaomukang/datasets/isic18/test"
    
    batch_size = 64
    num_epochs = 150



train_names = read_file_names(TRAIN_PATH)
train_gt_names = read_file_names(TRAIN_GT_PATH)
train_datas = build_dataset(train_names, train_gt_names)

split_resize_save_img(train_datas, TRAIN_SAVE_PATH)


test_names = read_file_names(TEST_PATH)
test_gt_names = read_file_names(TEST_GT_PATH)
test_datas = build_dataset(test_names, test_gt_names)

split_resize_save_img(test_datas, TEST_SAVE_PATH)