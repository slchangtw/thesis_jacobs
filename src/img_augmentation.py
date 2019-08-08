#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import sys

import numpy as np

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[0] - crop_height
    max_y = image.shape[1] - crop_width

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[x: x + crop_width, y: y + crop_height, :]

    return crop

def img_augmentation(path):
    
    files = [path+f for f in os.listdir(path) if f.endswith('.jpg')]
    
    for file in files:
        print('Processing {}'.format(file))
        img = cv2.imread(file)
        
        tmp = file.split('.')

        # flip horizontally
        cv2.imwrite(tmp[0] + '_f_h.' + tmp[1], cv2.flip(img, 0))

        # flip vertically
        cv2.imwrite(tmp[0] + '_f_v.' + tmp[1], cv2.flip(img, 1))

        # flip both
        cv2.imwrite(tmp[0] + '_f_b.' + tmp[1], cv2.flip(img, -1))

        # rotate 90
        M_90 = np.rot90(img, k=1)
        cv2.imwrite(tmp[0] + '_r_90.' + tmp[1], M_90)

        # rotate 270           
        M_270 = np.rot90(img, k=3)
        cv2.imwrite(tmp[0] + '_r_270.' + tmp[1], M_270)

        # crop
        height, width = img.shape[0], img.shape[1]
        if height < 1000:
            crop = get_random_crop(img, 640, 2048)
            cv2.imwrite(tmp[0] + '_crop.' + tmp[1], cv2.resize(crop, dsize=(width, height)))
        else:
            crop = get_random_crop(img, 2048, 2048)
            cv2.imwrite(tmp[0] + '_crop.' + tmp[1], cv2.resize(crop, dsize=(width, height)))
    
    return None

if __name__ == '__main__':
    path = sys.argv[1]
    img_augmentation(path)
    
    