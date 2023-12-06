import cv2
import re
import os

def resize_and_padding(img, target=256, is_3d=True):
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