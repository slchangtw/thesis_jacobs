#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from collections import Counter

def get_number_classes(path):
    
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    txt_files.remove('classes.txt')
    
    total_classes = []
    
    for file in txt_files:
        with open(path+file, 'r') as f:
            total_classes = total_classes + [x.split(' ')[0] for x in f.readlines()]
        
    class_counter = Counter(total_classes)
    
    classes_file = path + 'classes.txt'
    
    with open(classes_file, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
        mapping = {i:lines[i] for i in range(len(lines))}
    
    mapping = dict((mapping[int(key)], value) for (key, value) in class_counter.items())
    
    for key in sorted(mapping):
        print(key, mapping[key])
    
    return None

if __name__ == '__main__':
    path = sys.argv[1]
    get_number_classes(path)
