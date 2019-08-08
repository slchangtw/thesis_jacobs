#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

import sys

if __name__ == '__main__':
    data_dir = sys.argv[1]
    type_ = sys.argv[2]

    with open(data_dir + type_ + '.txt', 'w') as f:
        for path_and_filename in glob.iglob(os.path.join(data_dir + type_, '*.jpg')):  
            title, ext = os.path.splitext(os.path.basename(path_and_filename))
            f.write(data_dir + type_+ '/' + title + '.jpg' + '\n')
