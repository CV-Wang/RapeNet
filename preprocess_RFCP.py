import json
import numpy as np
import os
from PIL import Image
import cv2
import argparse
import shutil


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    point = np.array(point)
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def parseJson(Json):
    with open(Json) as f:
      jsonlist = json.load(f)
      bbs = []
      for jsondict in jsonlist['step_1']['result']:
          cx = int(jsondict['x'])
          cy = int(jsondict['y'])
          bbs.append([cx,cy, cx,cy])
    return bbs


def bbs2points(bbs):
    points = []
    m = []
    for bb in bbs:
        x1, y1, x2, y2 = [float(b) for b in bb]
        x, y = np.round((x1 + x2) / 2).astype(np.int32), np.round((y1 + y2) / 2).astype(np.int32)
        points.append([x/2, y/2])
    return points


def generate_data(images_dir, labels_dir, label_name):
    im = Image.open(images_dir + label_name.split('.')[0] + '.png')
    im_w, im_h = im.size
    bbs = parseJson(labels_dir + label_name)
    points = bbs2points(bbs)
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), bbs2points(bbs), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='datasets/RFCP',
                        help='original data directory')
    parser.add_argument('--data-dir', default='datasets/RFCP_processed',
                        help='processed data directory')
    args = parser.parse_args()
    return args

def FindAllFiles(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


if __name__=='__main__':
    args = parse_args()
    save_dir = args.data_dir
    data_dir = args.origin_dir

    list = ['train', 'val', 'test']

    rootdir = './datasets/RFCP_processed'
    filelist = []
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(filepath+" removed!")
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
            print("dir "+filepath+" removed!")

    if not os.path.exists("./datasets/RFCP_processed"):
        os.makedirs("./datasets/RFCP_processed")
    if not os.path.exists("./datasets/RFCP_processed/train"):
        os.makedirs("./datasets/RFCP_processed/train")
    if not os.path.exists("./datasets/RFCP_processed/val"):
        os.makedirs("./datasets/RFCP_processed/val")
    if not os.path.exists("./datasets/RFCP_processed/test"):
        os.makedirs("./datasets/RFCP_processed/test")

    min_size = 512
    max_size = 1024


    for phase in list:
        images_dir = os.path.join(args.origin_dir, phase) + '/images/'
        labels_dir = os.path.join(args.origin_dir, phase) + '/labels/'
        sub_save_dir = os.path.join(args.data_dir, phase) + '/'

        # 调用方法
        for label in FindAllFiles(labels_dir):
            label_name = label.split('/')[4]
            print(label_name)
            image, _, points = generate_data(images_dir, labels_dir, label_name)

            if phase == 'train' or phase == 'val':
                # rape_flowers: train & val need
                dis = find_dis(points)
                points = np.concatenate((points, dis), axis=1)

            image_save_name = label_name.split('.')[0] + '.png'
            im_save_path = os.path.join(sub_save_dir, image_save_name)
            image.save(im_save_path)

            gd_save_path = im_save_path.replace('png', 'npy')
            np.save(gd_save_path, points)
