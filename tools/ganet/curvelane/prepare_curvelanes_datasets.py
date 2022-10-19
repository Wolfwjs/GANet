"""
    convert Curvelanes dataset to CULane formal
"""
import argparse
import json
import os

from tqdm import tqdm
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Curvelanes dataset root path')
    args = parser.parse_args()
    return args


def prepare_labels(src, dst):
    label_files = glob(os.path.join(src,'*.json')) # 100000个
    for label_file in tqdm(label_files):
        with open(label_file, 'r') as json_file:
            json_dict = json.load(json_file)
        file_name = os.path.splitext(os.path.basename(label_file))[0] # '52c9a73e35cda4cd82080d84582de4d9.lines'
        save_name = '{}.txt'.format(file_name) # '52c9a73e35cda4cd82080d84582de4d9.lines.txt'
        lines = json_dict['Lines']
        coord_lines = []
        for line in lines:
            coord_line = []
            for coord in line:
                coord_x = coord['x']
                coord_y = coord['y']
                coord_line.append(coord_x)
                coord_line.append(coord_y)
            coord_lines.append(coord_line)
        save_dir = os.path.join(dst, save_name)
        with open(save_dir, 'w') as write_f:
            for idx, coord_line in enumerate(coord_lines):
                for coord in coord_line:
                    print(coord, file=write_f, end=' ')
                if idx != len(coord_lines) - 1:
                    print(file=write_f)

def main():
    args = parse_args()
    train_src = os.path.join(args.path, 'train/labels')
    train_dst = os.path.join(args.path, 'train/images')
    valid_src = os.path.join(args.path, 'valid/labels')
    valid_dst = os.path.join(args.path, 'valid/images')
    prepare_labels(train_src, train_dst)
    prepare_labels(valid_src, valid_dst)


if __name__ == '__main__':
    main()