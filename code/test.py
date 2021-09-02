# -*- coding: utf-8 -*-
# @Author : Linkin Lai
# @Time : 2021/9/2 16:27
# @File : test.py.py
# @Project : ILDNet
# @CopyRight: AI Lab 2021
# @Description: ILDNet的测试代码

# import here
from data import Dataset2D
import torch
import time
import copy
from torch.utils.data import Dataset, DataLoader
from model import Unet
from collections import defaultdict
from torchsummary import summary
from torchvision import transforms
import os
import numpy as np


#####################################################准备数据集#############################################################

# 用于测试的简易数据集
class SimpleDataset(Dataset):

    def __init__(self, num: 'int' = 100, transform: 'function' = None) -> None:
        super().__init__()
        # self.data,self.label = torch.random((num,1,512,512))
        self.data = [np.random.rand(1, 512, 512)] * num
        self.label = [np.random.rand(1, 512, 512)] * num
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: 'int') -> list:
        data = self.data[item]
        label = self.label[item]
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)

        return data, label


################################################定义测试过程##############################################################


def test_model(
        model: 'torch.nn.Module',
        model_state_dict_path: 'str',
        test_dataset: 'torch.utils.data.dataset',
        pre_img_path: 'str'
):
    batch_size = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 打印模型框架
    summary(model, input_size=(1, 512, 512))
    # 加载模型
    print('loading model state dict')
    model.load_state_dict(torch.load(model_state_dict_path, map_location=lambda storage, loc: storage))
    # 加载测试数据
    autoloader = DataLoader(test_dataset, batch_size, shuffle=False)
    total_len = len(test_dataset)
    index = 1
    # 进行数据测试并保存
    for inputs, labels in autoloader:
        print('\rTesting the data: {}/{}'.format(index*batch_size, total_len), end='', flush=True)
        # 初始化inputs和labels的设置
        inputs = inputs.float().to(device)
        output = model(inputs)
        save(output, pre_img_path, batch_size,index)
        index += 1
    print('\nTest is over!')


def save(
        img: 'torch.tensor',
        save_path: 'str',
        batch_size: 'int',
        index: 'int'
):
    for _ in range(batch_size):
        new_img = transforms.ToPILImage()(img[_]).convert('RGB')
        result_img_path = os.path.join(save_path, '{}.jpg'.format(str(_ + 1 + (index - 1) * batch_size).zfill(5)))
        new_img.save(result_img_path)


if __name__ == '__main__':
    """
    sim_dataset = SimpleDataset(10)
    model_path = '.././experiment/model_pkl/model1.pkl'
    result_path = '.././experiment/test_result'
    model = Unet(1)
    test_model(model, model_path, sim_dataset, result_path)
    """

    dataset =