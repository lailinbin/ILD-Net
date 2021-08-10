# -*- coding: UTF-8 -*-
'''
#Data: 2021/8/10 20:00
#FilePath: data_statistics.py
#Author: LaiLinbin
#CopyRight: AI Lab 2021
#Description: This file is used to calculate the statistics information of dataset
'''
# import lib here
from data import Data_statistics
import argparse


def get_parser():
    # 实例化ArgumentParser
    parser = argparse.ArgumentParser(description='please input the path of dataset')
    # 添加参数设置
    parser.add_argument('-i',
                        type=str,
                        dest='dataset_path',
                        default='.././3Dimg_data',
                        help='path of dataset for statistics analysis')

    parser.add_argument('-o',
                        type=str,
                        dest='excel_path',
                        default='.././experiment/statistics.xlsx',
                        help='path of excel file to save the statistics information')

    argvs = parser.parse_args()
    return argvs


if __name__ == '__main__':
    args = get_parser()
    data_statistics = Data_statistics(args.dataset_path)
    print('统计样本信息中')
    data_statistics.save_statistics_information(args.excel_path)
    print('样本信息统计完成')
    pass
