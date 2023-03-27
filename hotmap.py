import os
import cv2
import glob
import h5py
import scipy
import numpy as np
from itertools import islice
from tqdm import tqdm
from sortedcontainers import SortedDict
from scipy.ndimage.filters import gaussian_filter
import json
import xml.etree.ElementTree as ET

def get_img_pathes(path_sets):
    """
    Return all images from all pathes in 'path_sets'
    """
    img_pathes = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_pathes.append(img_path)
    return img_pathes


def save_computed_density(density_map, out_path):
    """
    Save density map to h5py format
    """
    with h5py.File(out_path, 'w') as hf:
        hf['density'] = density_map


def compute_sigma(gt_count, distance=None, min_sigma=1, method=1, fixed_sigma=15):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (sum of distance to 3 nearest neighbors) / 10
    * method = 2 : sigma = distance to nearest neighbor
    * method = 3 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    """
    if gt_count > 1 and distance is not None:
        if method == 1:
            sigma = np.mean(distance[1:4]) * 0.1
        elif method == 2:
            sigma = distance[1]
        elif method == 3:
            sigma = fixed_sigma
    else:
        sigma = fixed_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return sigma


def find_closest_key(sorted_dict, key):
    """
    Find closest key in sorted_dict to 'key'
    """
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


def gaussian_filter_density(non_zero_points, map_h, map_w, distances=None, kernels_dict=None, min_sigma=2, method=1,
                            const_sigma=15):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    # gt_count = non_zero_points.shape[0]
    gt_count=len(non_zero_points)
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        sigma = compute_sigma(
            gt_count, distances[i], min_sigma=min_sigma, method=method, fixed_sigma=const_sigma)
        closest_sigma = find_closest_key(kernels_dict, sigma)
        kernel = kernels_dict[closest_sigma]
        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2

        min_img_x = max(0, point_x - kernel_size)
        min_img_y = max(0, point_y - kernel_size)
        max_img_x = min(point_x + kernel_size + 1, map_h - 1)
        max_img_y = min(point_y + kernel_size + 1, map_w - 1)

        kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
        kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        density_map[min_img_x:max_img_x,
        min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    return density_map


def get_gt_dots(mat_path):
    """
    Load Matlab file with ground truth labels and save it to numpy array.
    ** cliping is needed to prevent going out of the array
    """
    filetype = mat_path.split('.')[2]
    points = []
    bbs = []
    if filetype == 'xml':
            tree = ET.parse(mat_path)
            root = tree.getroot()
            for bb in root.iter('bndbox'):
                xmin = int(bb.find('xmin').text)
                ymin = int(bb.find('ymin').text)
                xmax = int(bb.find('xmax').text)
                ymax = int(bb.find('ymax').text)
                bbs.append([xmin, ymin, xmax, ymax])
                x, y = np.round((xmin + xmax) / 2).astype(np.int32), np.round((ymin + ymax) / 2).astype(np.int32)
                points.append([x, y])

    elif filetype == 'json':
            with open(mat_path) as f:
              jsonlist = json.load(f)
            for jsonitem in jsonlist:
                for jsondict in jsonlist[jsonitem]['regions']:
                    type = jsondict['shape_attributes']['name']
                    if type == 'rect':
                        x = int(jsondict['shape_attributes']['x'])
                        y = int(jsondict['shape_attributes']['y'])
                        width = int(jsondict['shape_attributes']['width'])
                        height = int(jsondict['shape_attributes']['height'])
                        xmax = x + width
                        ymax = y + height
                        bbs.append([x, y, xmax, ymax])
                        x, y = np.round((x + xmax) / 2).astype(np.int32), np.round((y + ymax) / 2).astype(np.int32)
                        points.append([x, y])
                    elif type == 'point':
                        cx = int(jsondict['shape_attributes']['cx'])
                        cy = int(jsondict['shape_attributes']['cy'])
                        bbs.append([cx, cy, cx, cy])
                        x, y = np.round(cx).astype(np.int32), np.round(cy).astype(np.int32)
                        points.append([x, y])

    return points


def set_circles_on_img(image, bbox_list, circle_size=2):
    """
    Set circles on images at centers of bboxes in bbox_list
    """
    for bbox in bbox_list:
        cv2.circle(image, (bbox[0], bbox[1]), circle_size, (255, 0, 0), -1)
    return image


def generate_gaussian_kernels(round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=20, num_sigmas=801):
    """
    Computing gaussian filter kernel for sigmas in linspace(sigma_min, sigma_max, num_sigmas) and saving
    them to dict.
    """
    kernels_dict = dict()
    sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
    for sigma in tqdm(sigma_space):
        sigma = np.round(sigma, decimals=round_decimals)
        kernel_size = np.ceil(sigma * sigma_threshold).astype(np.int)

        img_shape = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = scipy.ndimage.filters.gaussian_filter(
            arr, sigma, mode='constant')
        kernel = arr / arr.sum()
        kernels_dict[sigma] = kernel

    return kernels_dict


if __name__ == '__main__':
    # uncomment to generate and save dict with kernel sizes
    kernels_dict = generate_gaussian_kernels(round_decimals=3,
                                             sigma_threshold=4, sigma_min=0, sigma_max=20, num_sigmas=801)

    kernels_dict = SortedDict(kernels_dict)

    map_out_folder = 'maps_fixed_kernel/'
    min_sigma = 2
    method = 3
    const_sigma = 15