# -*- coding: UTF-8 -*-
'''
#Data: 2021/8/10 20:49
#FilePath: data_preprocessor
#Author: LaiLinbin
#CopyRight: AI Lab 2021
#Description: 
'''
# import lib here
from data import Data_preprocessor
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='To preprocess the dataset. 1. rename dicom file; 2.create 2D Dataset '
                                                 'from original dataset')
    parser.add_argument('-original_img_dir',
                        type=str,
                        dest='original_img_dir',
                        default='.././3Dimg_data',
                        help='path of original dataset')
    parser.add_argument('-np_data_dir',
                        type=str,
                        dest='np_data_dir',
                        default='.././2Dimg_data',
                        help='path of 2D npy dataset')
    argvs = parser.parse_args()
    return argvs


if __name__ == '__main__':
    args = get_args()
    original_img_dir = args.original_img_dir
    np_data_dir = args.np_data_dir

    data_preprocessor = Data_preprocessor(original_img_dir, np_data_dir)
    data_preprocessor.original_dicomfile_rename()
    data_preprocessor.data_and_label_generator()
    pass
