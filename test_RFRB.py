import torch
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
import numpy as np
import os
from os import listdir
import cv2
import argparse
import scipy
import scipy.stats
from PIL import Image
from hotmap import generate_gaussian_kernels, gaussian_filter_density
from sortedcontainers import SortedDict
import scipy.spatial as T
from datasets.rape import Rape

from models.RapeNet import net
# from models.RapeNet_plus import net


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='datasets/RFRB_processed',
                        help='training data directory')
    parser.add_argument('--origin-dir', default='datasets/RFRB/test',
                        help='origin data directory')
    parser.add_argument('--save-dir', default='ckpt/RFRB/RapeNet',
                        help='model directory, ckpt/RFRB/RapeNet, ckpt/RFRB/RapeNet_plus')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args

def parseJson(Json):
    with open(Json) as f:
      jsonlist = json.load(f)
    bbs = []
    for jsonitem in jsonlist:
     for jsondict in jsonlist[jsonitem]['regions']:
        type = jsondict['shape_attributes']['name']
        if type == 'rect':
            x = int(jsondict['shape_attributes']['x'])
            y = int(jsondict['shape_attributes']['y'])
            width= int(jsondict['shape_attributes']['width'])
            height = int(jsondict['shape_attributes']['height'])
            xmax= x+width
            ymax= y+height
            bbs.append([x, y, xmax, ymax])
        elif type == 'point':
           cx = int(jsondict['shape_attributes']['cx'])
           cy = int(jsondict['shape_attributes']['cy'])
           bbs.append([cx,cy, cx,cy])
    return bbs

def parsexml(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    bbs = []
    for bb in root.iter('bndbox'):
        xmin = int(bb.find('xmin').text)
        ymin = int(bb.find('ymin').text)
        xmax = int(bb.find('xmax').text)
        ymax = int(bb.find('ymax').text)
        bbs.append([xmin, ymin, xmax, ymax])
    return bbs

def bbs2points(bbs):
    points = []
    m = []
    for bb in bbs:
        x1, y1, x2, y2 = [float(b) for b in bb]
        x, y = np.round((x1 + x2) / 2).astype(np.int32), np.round((y1 + y2) / 2).astype(np.int32)
        points.append([x, y])
    return points

def generate_data(images_dir, labels_dir, label_name):
    im = Image.open(images_dir + label_name.split('.')[0] + '.png')
    im_w, im_h = im.size
    filetype = label_name.split('.')[1]

    if filetype == 'xml':
        bbox = parsexml(labels_dir + label_name)
    elif filetype == 'json':  # json
        bbox = parseJson(labels_dir + label_name)
    points = bbs2points(bbox)
    return points


def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae


def compute_rmse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    rmse = np.sqrt(np.mean((diff ** 2)))
    return rmse

def compute_relerr(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        rmae = np.mean(np.abs(diff) / gt) * 100
        rrmse = np.sqrt(np.mean(diff**2 / gt**2)) * 100
    else:
        rmae = 0
        rrmse = 0
    return rmae, rrmse


def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    pd, gt = np.array(pd), np.array(gt)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
    return r_value**2


def FindAllFiles(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_dir + '/images'):
        os.makedirs(args.save_dir + '/images')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Rape(os.path.join(args.data_dir, 'test'), 256, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = net()

    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []
    epoch_gt = []
    epoch_pd = []
    save_outs = []
    cmap = plt.cm.get_cmap('jet')

    with open(args.save_dir + '/eval.txt', 'a') as f:
        for inputs, count, name in dataloader:
            inputs = inputs.to(device)

            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                temp_minu = count[0].item() - torch.sum(outputs).item()
                gt = count[0].item()
                pd = torch.sum(outputs).item()

                insz = 256
                os = 4

                image_name, = name
                images_dir = args.origin_dir + '/images/'
                labels_dir = args.origin_dir + '/labels/'

                for label in listdir(labels_dir):

                    labels_name = label.split('.')[0]

                    if labels_name == image_name:
                        label_name = label

                        im = cv2.imread(images_dir + image_name + '.png')
                        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        image1 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        image2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                        m = image2.shape
                        h = m[0]
                        w = m[1]

                        save_im_dir = args.save_dir + '/images'
                        outputs_show = outputs.squeeze().detach().cpu().numpy()
                        outputs_show = (outputs_show - outputs_show.min()) / (outputs_show.max() - outputs_show.min() + 1e-5)

                        inh, inw = outputs_show.shape[:2]
                        size = (inw, inh)

                        outputs_show = cmap(outputs_show)
                        outputs_show = outputs_show[:, :, 0:3]*255
                        outputs_show = cv2.resize(outputs_show, (w, h), interpolation=cv2.INTER_AREA)
                        outputs_show = 0.6 * image2 + 0.4 * outputs_show

                        outputs_show = np.trunc(outputs_show)

                        points_origin = generate_data(images_dir, labels_dir, label_name)

                        for i in list(points_origin):
                            cv2.circle(image, (i[0], i[1]), 6, (255, 0, 0), -1)

                        nh = h
                        nw = w
                        kernels_dict = generate_gaussian_kernels(round_decimals=1, sigma_threshold=10, sigma_min=0, sigma_max=20, num_sigmas=801)
                        kernels_dict = SortedDict(kernels_dict)
                        min_sigma = 2
                        method = 3
                        const_sigma = 5
                        tree = T.KDTree(points_origin.copy(), leafsize=1024)  # build kdtree
                        distances, _ = tree.query(points_origin, k=1)  # query kdtree
                        target = gaussian_filter_density(points_origin, nh, nw, distances, kernels_dict, min_sigma=min_sigma, method=method, const_sigma=const_sigma)
                        target = target / (target.max() + 1e-12)
                        target = cmap(target) * 255
                        target_show = 0.6 * image1 + 0.4 * target[:, :, 0:3]
                        target_show = np.trunc(target_show)
                        manual_image = image

                        fig = plt.figure()
                        ax1 = fig.add_subplot(1, 3, 1)
                        ax1.imshow(manual_image.astype(np.uint8))
                        ax1.set_title('manual count=%4.2f' % gt)
                        ax1.get_xaxis().set_visible(False)
                        ax1.get_yaxis().set_visible(False)
                        ax2 = fig.add_subplot(1, 3, 2)
                        ax2.imshow(target_show.astype(np.uint8))
                        ax2.set_title('groundtruth heatmap')
                        ax2.get_xaxis().set_visible(False)
                        ax2.get_yaxis().set_visible(False)
                        ax3 = fig.add_subplot(1, 3, 3)
                        ax3.imshow(outputs_show.astype(np.uint8))
                        ax3.set_title('inferred count=%4.2f' % pd)
                        ax3.get_xaxis().set_visible(False)
                        ax3.get_yaxis().set_visible(False)

                        plt.tight_layout(rect=[0, 0, 1, 1.2])  # rape flower counting
                        plt.savefig('{}/{}.png'.format(save_im_dir, image_name), bbox_inches='tight', dpi=300)
                        plt.close()

                save_out = (image_name + '.png  ', round(temp_minu, 2), count[0].item(), round(torch.sum(outputs).item(), 2))
                save_out = ' '.join(str(i) for i in save_out)
                print(save_out)
                print(save_out, file=f)
                save_outs.append(save_out)
                epoch_gt.append(gt)
                epoch_pd.append(pd)

        mae = compute_mae(epoch_pd, epoch_gt)
        rmse = compute_rmse(epoch_pd, epoch_gt)
        rmae, rrmse = compute_relerr(epoch_pd, epoch_gt)
        r2 = rsquared(epoch_pd, epoch_gt)

        log_str = 'Final_compute1: mae {}, rmse {}, rmae {}, rrmse {}, r2 {}'.format(mae, rmse, rmae, rrmse, r2)

        print(log_str)
        print(log_str, file=f)
