# coding=utf-8
'''
#Date: 2021-08-03 16:21:37
#LastEditors: Lai Linbin
#LastEditTime: 2021-08-06 22:12:40
#FilePath: \project\code\data.py
#CopyRight: AI Lab 2020
#Description: ILD 项目的数据预处理程序，包含数据统计、数据增强、数据库制作。。。
'''
#import here
import os
from typing import Dict 
import numpy as np
from numpy.core.defchararray import index, zfill
from numpy.core.fromnumeric import shape
import pydicom
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib import animation
import pandas as pd
import torch
from torch.utils.data import DataLoader, dataloader


class Data_statistics():

    '''
    #description: Data_statistics类用于数据集的信息统计，包括样本数量，CT图层数量，各表征标签图层数量等
    #param {class} self: 类自身，固定参数
    #param {str} data_dir: dicom数据和标签存放文件夹路劲
    #return {None}
    '''
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        pass

    '''
    #description: 从dicom文件中统计每个病例含有的CT图片数量
    #param {class} self：类自身，固定参数
    #param {int} start_index：开始索引值，默认为0
    #return {Dict}：样本统计信息存储的字典，格式为{'样本编号':该样本含有图片数量}
    '''
    def calculate_dicom(self,start_index = 0) -> Dict:
        sample_index = start_index# 样本编号
        sample_statistics_dict={}# 样本统计信息存储的字典，格式为{'样本编号':该样本含有图片数量}

        for sample_dir in os.listdir(self.data_dir):
            dicom_dir = os.path.join(self.data_dir,sample_dir,'dicom')
            img_num = len(os.listdir(dicom_dir)) #图片层数就是文件夹下的文件数量，也就是列表长度
            sample_statistics_dict[sample_index] = img_num 
            sample_index += 1
        
        return sample_statistics_dict

    '''
    #description: 从nii文件中统计每个病例含有的各种标签的数量
    #param {class} self：类自身，固定参数
    #param {int} start_index：开始的索引值，默认为0
    #return {Dict}：样本统计信息存储的字典，格式为{'样本编号':该样本各标签数量列表}
    '''
    def calculate_nii(self,start_index = 0) -> Dict:
        sample_index = start_index # 样本编号
        sample_statistics_dict = {} #样本统计信息存储的字典，格式为{'样本编号':该样本各标签数量列表}
        sample_dirs = os.listdir(self.data_dir)
        total_sample_nums = len(sample_dirs)

        for sample_dir in sample_dirs:
            nii_dir = os.path.join(self.data_dir,sample_dir,'result')
            nii_statistics_list = [0 for x in range(10)]
            for nii_filename in os.listdir(nii_dir):
                nii_path = os.path.join(nii_dir,nii_filename)
                # print(nii_filename[5:-4])
                list_index = int(nii_filename[5:-4])-1 # 获取标签编号，由于标签是从1开始，而列表索引值是从0开始，所以需要减去1
                nii_label = nib.load(nii_path) # 加载nii文件
                label_data = nii_label.get_fdata()
                label_num = self.calculate_label_num(np.array(label_data)) #使用self.calculate_label_num函数统计nii中带有标签的图片数
                nii_statistics_list[list_index] = label_num
            
            sample_statistics_dict[sample_index] = nii_statistics_list
            sample_index += 1
            
            print(f'已处理{sample_index}份数据，共有{total_sample_nums}份数据')


        
        return sample_statistics_dict
    
    '''
    #description: 统计传入的nii数据中含有标签的图层数量
    #param {class} self：类自身，固定参数
    #param {np.array} label_data：转换成np.array格式的nii数据
    #return {int}: 含有有效标签的图层数量
    '''   
    def calculate_label_num(self,label_data: np.array) -> int:
        label_num = 0
        for label_index in range(label_data.shape[2]):
            if np.sum(label_data[:,:,label_index]) != 0:#含有有效标签则矩阵合不为0
                label_num += 1
            else:
                pass
        return label_num
    
    '''
    #description: 
    #param {class} self：类自身，固定参数
    #param {int} label_cls：标签类别的数量，默认为10
    #param {int} start_index：计算dicom和nii统计数据时需要传入的开始索引值，默认为0，详情可看方法calculate_dicom、calculate_nii
    #return {pd.Dateframe}: 将统计信息以pd.Dataframe的格式返回，详细统计信息为data_statistics_df_detail，概述信息为data_statistics_df_summary
    '''    
    def build_Dataframe(self,label_cls = 10, start_index = 0) -> pd.DataFrame:
        # 获取dicom和nii的统计信息，即dicom_dict和nii_dict
        dicom_dict = self.calculate_dicom(start_index)
        nii_dict = self.calculate_nii(start_index)
        
        # 创建两个字典分别存储详细统计信息（data_statistics_dict）和简略信息（data_statistics_dict_summary）
        data_statistics_dict = nii_dict 
        data_statistics_dict_summary = {}

        # 将dicom的统计信息整合到data_statistics_dict上
        for key in dicom_dict.keys():
            data_statistics_dict[key].append(dicom_dict[key])

        data_statistics_df_detail = pd.DataFrame(nii_dict)# 存储样本详细信息的Dataframe

        # 用summary_list存储信息
        summary_list1 = [0 for x in range(11)]# 存储各个标签图层总数
        summary_list2 = [0 for x in range(11)]# 存储各个标签图层占比
        # 将各个标签图层总数的统计信息存储到data_statistics_dict_summary上
        for key in data_statistics_dict.keys():
            for i in range(11):
                summary_list1[i] += data_statistics_dict[key][i]
        # 将各个标签图层占比的统计信息存储到data_statistics_dict_summary上
        for i,num in enumerate(summary_list1):
            summary_list2[i] = num/summary_list1[-1]
        
        # 将summary_list的信息整合到data_statistics_dict_summary中
        data_statistics_dict_summary['total_num']=summary_list1
        data_statistics_dict_summary['proportion'] = summary_list2
        data_statistics_df_summary = pd.DataFrame(data_statistics_dict_summary)
        return data_statistics_df_detail, data_statistics_df_summary

    '''
    #description: 
    #param {class} self: 类自身，固定参数
    #param {str} excel_path: 统计信息保存的excel文件夹路径
    #param {int} label_cls: 标签种类，默认为10
    #param {int} start_index: 进行样本统计时的初始病例索引值，默认为0
    #return {None}
    '''
    def save_statistics_information(self,excel_path:str,label_cls=10,start_index=0) -> None:
        detail_df,summary_df = self.build_Dataframe(label_cls,start_index)
        with pd.ExcelWriter(excel_path) as writter:
            detail_df.to_excel(writter,sheet_name='detail')
            summary_df.to_excel(writter,sheet_name='summary')

class Data_preprocessor():
    
    '''
    #description: 对数据进行预处理，包括dicom文件重命名，npy格式数据生成，npy文件可视化等。如果是由fact导出的数据，
    # 第一次生成3D数据库需要先进行dicom文件重命名。生成2D数据库则需要进行npy格式数据生成。npy文件可视化用于确认生成
    # 的npy文件数据与标签是否匹配
    #param {class} self: 类自身，固定参数
    #param {str} original_img_dir: fact软件导出的数据文件夹路径
    #param {str} np_data_dir：npy文件保存路径
    #return {None}
    '''    
    def __init__(self,original_img_dir:str,np_data_dir:str):
        self.original_img_dir = original_img_dir
        self.np_data_dir = np_data_dir

    '''
    #description: 给从fact软件导出的dicom文件进行重命名。由于fact文件导出的dicom文件数字长度不同，如果不进行重命名后续处理会出现顺序混乱，与标签不匹配
    #param {class} self: 类自身，固定参数
    #return {type param}
    '''
    def original_dicomfile_rename(self) -> None:
        index = 1
        #遍历original_img_dir文件夹下各个病例
        sample_list = os.listdir(self.original_img_dir)
        for sample in sample_list:
            #遍历样本下的dicom文件
            for dicom_file_name in os.listdir(os.path.join(self.original_img_dir,sample,'dicom')):
                src_dicom_filename = os.path.join(self.original_img_dir,sample,"dicom",dicom_file_name)
                dst_dicom_filename = os.path.join(self.original_img_dir,sample,"dicom",dicom_file_name[:-4].zfill(5)+dicom_file_name[-4:])
                if not os.path.exists(dst_dicom_filename):
                    os.rename(src=src_dicom_filename,dst=dst_dicom_filename)
            print(f"{sample} has been rename successful!\nprocessing：\t{index/len(sample_list)}")
            index += 1
    
    '''
    #description: 从dicom文件读取数据并转换为np.array格式的数据并返回
    #param {class} self：类自身，固定参数
    #param {str} dicom_path：dicom文件的路径
    #return {np.ndarray}: dicom文件中提取的数据
    '''    
    def dicom_to_np(self, dicom_path:str) -> np.ndarray:
        dicom_img = pydicom.dcmread(dicom_path)
        return np.array(dicom_img.pixel_array)

    '''
    #description:将dicom和nii文件转换成ndarray格式的数据保存在2Dimg_data文件夹下 
    #param {class} self： 类自身固定参数
    #return {type param}
    '''
    def data_and_label_generator(self) -> None:
        # 设置进度条索引值progression_index、病例索引值sample_index
        progression_index = 0
        sample_index = 0

        #获取病例列表
        sample_list = os.listdir(self.original_img_dir)
        total_progression = len(sample_list)

        # 遍历病例
        for sample in sample_list:
            # 遍历nii文件（即label）,如果有label才导出dicom的np.ndarray
            for nii_filename in os.listdir(os.path.join(self.original_img_dir,sample,'result')):
                # 获取nii文件路径
                nii_path = os.path.join(self.original_img_dir,sample,'result',nii_filename)
                # 获取标签类别
                label_class = nii_filename[:-4]
                # 读取nii文件，获取标签信息
                nii_label = nib.load(nii_path)
                label_array = np.array(nii_label.get_fdata())
                # 遍历标签信息，将含有标注信息的标签提取出来
                for label_index in range(label_array.shape[2]):
                    # 判断标签是否存在,如果存在则进行保存操作
                    if np.sum(label_array[:,:,label_index]) != 0:
                        # 设置np.ndarray的保存位置，文件名格式为：样本编号_层数编号.npy。如第一个病例第54层，则文件名为：001_00054.npy
                        np_filename = str(sample_index).zfill(4)+'_'+str(label_index).zfill(5)+'.npy'
                        np_label_path = os.path.join(self.np_data_dir,"label",label_class,np_filename)

                        # 检查文件夹是否存在，如果不存在则自动建一个新文件夹
                        if not os.path.exists(os.path.join(self.np_data_dir,"label",label_class)):
                            os.mkdir(os.path.join(self.np_data_dir,"label",label_class))
                        # 保存标签信息
                        np.save(np_label_path,label_array[:,:,label_index].T)

            # 读取dicom文件并保存data数据
            for dicom_filename in os.listdir(os.path.join(self.original_img_dir,sample,'dicom')):
                # 获取dicom文件路径（dicom_path）和数据的np.ndarray保存路劲(np_data_path)
                dicom_path = os.path.join(self.original_img_dir,sample,'dicom',dicom_filename)
                np_data_filename = str(sample_index).zfill(4) + '_' + dicom_filename[:-4].zfill(5) + '.npy'
                np_data_path = os.path.join(self.np_data_dir,"data",np_data_filename)

                #检测文件夹是否存在，如果不存在则自动建立一个新文件夹
                if not os.path.exists(os.path.join(self.np_data_dir,"data")):
                    os.mkdir(os.path.join(self.np_data_dir,"data"))

                #读取dicom文件并保存npy文件
                data_ndarray = self.dicom_to_np(dicom_path)
                np.save(np_data_path,data_ndarray)
            
            # 更新样本编号打印进度条
            sample_index += 1
            progression_index += 1
            print(f"saving nii label\tprocessing:{sample_index/total_progression}")

    '''
    #description: 可视化nparray文件数据以检测标签与CT图像是否匹配
    #param {class} self: 类自身，固定参数
    #param {int} label_index：标签类别，可以查看文件夹./2Dimg_data/label获取，如果要查看label1文件夹下的标签，则label_index=1
    #return {None}
    '''
    def data_check_and_show(self,label_index:int):

        # 获取label保存的文件夹路径
        label_dir_path = os.path.join(self.np_data_dir,'label','label' + str(label_index))
        # 判断标签是否存在，如果不存在则输出提示并退出
        if  not os.path.exists(label_dir_path):
            print(f'label{label_index} is not exists! Please check the label_dir to ensure the label index.')
            return None
        
        # 遍历读取label文件并进行可视化显示。
        for file_name in os.listdir(label_dir_path):
            # 设定data和label的文件路径
            label_path = os.path.join(label_dir_path,file_name)
            data_path = os.path.join(self.np_data_dir,'data',file_name)

            # 加载npy文件
            data = np.load(data_path)
            label = np.load(label_path)

            # 可视化显示
            fig = plt.figure()
            plt.imshow(data)
            plt.imshow(label,alpha=0.2,cmap='Reds')
            plt.show()

class Dataset2D(torch.utils.data.Dataset):
    
    '''
    #description: 用于单分类的npy二维数据集，通过label_index指定类别，从而提取特定类别的数据集
    #param {class} self: 类自身，固定参数
    #param {str} dataset_path：npy数据集存放路劲
    #param {int} label_index： 标签类别，可以查看文件夹./2Dimg_data/label获取，如果要查看label1文件夹下的标签，则label_index=1
    #return {None}
    '''    
    def __init__(self,dataset_path:str,label_index:int) -> None:
        super().__init__()
        self.data_dir = os.path.join(dataset_path,'data')
        self.label_dir = os.path.join(dataset_path,'label',f'label{label_index}')
        self.data_list = os.listdir(self.label_dir)

    def __getitem__(self,index:int) -> np.ndarray:
        # 获取data和label保存的文件路径
        data_filename = os.path.join(self.data_dir,self.data_list[index])
        label_filename = os.path.join(self.label_dir,self.data_list[index])

        # 读取文件获得npy数据
        data = np.load(data_filename)
        label = np.load(label_filename)

        return data.astype(float),label.astype(float)

    def __len__(self) -> int:
        return len(self.data_list)

class Dataset2D_multiply(torch.utils.data.Dataset):

    '''
    #description: 用于多分类的2D数据集，需要指定标签具有label_num个类别
    #param {class} self：类自身，固定参数
    #param {str} dataset_path：2D数据集存放位置
    #param {int} label_num：标签具有的类别个数，在本项目中为10
    #return {None}
    '''
    def __init__(self, dataset_path:str, label_num:int) -> None:
        super().__init__()
        self.data_dir = os.path.join(dataset_path,'data')
        self.label_dir = os.path.join(dataset_path,'label')
        self.data_list = os.listdir(self.data_dir)
        self.label_num = label_num

    def __getitem__(self,index:int) -> np.ndarray:
        # 获取data
        data_filename = os.path.join(self.data_dir,self.data_list[index])
        data = np.load(data_filename)

        print(data.shape)

        # 获取label
        label = []
        for class_index in range(self.label_num):
            label_path = os.path.join(self.label_dir,f'label{class_index+1}',self.data_list[index])
            if not os.path.exists(label_path):
                label.append(np.zeros(data.shape))
            else:
                sublabel = np.load(label_path)
                label.append(sublabel)
        
        return data,np.array(label)
    
    def __len__(self) -> int:
        return len(self.data_list)
            
class Dataset3D_from_fact(torch.utils.data.Dataset):

    '''
    #description: 用于单分类的3D数据集，通过label_index指定类别，从而提取特定类别的数据集
    #param {class} self: 类自身，固定参数
    #param {str} dataset_path：fact软件导出的数据集路径
    #param {int} label_index：标签类别，可以查看文件夹./2Dimg_data/label获取，如果要查看label1文件夹下的标签，则label_index=1
    #return {type param}
    '''
    def __init__(self,dataset_path:str,label_index:int) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.data_list = os.listdir(self.dataset_path)
        self.label_index = label_index

    def __getitem__(self,index:int) -> np.ndarray:
        # 获取data
        sample_dir = os.path.join(self.dataset_path,self.data_list[index])
        data_dir = os.path.join(sample_dir,'dicom')
        data = self.readdicom(data_dir)

        # 获取label
        label_dir = os.path.join(sample_dir,'result')
        label_path = os.path.join(label_dir,f'label{self.label_index}.nii')

        # 判断label是否存在，若不存在则返回0矩阵
        if not os.path.exists(label_path):
            label = np.zeros(data.shape)
        else:
            nii_label = nib.load(label_path)
            label_original = np.array(nii_label.get_fdata())
            label = self.transform_label(label_original)
        return data,label

    def __len__(self):
        return len(self.data_list)

    
    def readdicom(self,dicom_dir:str) -> np.ndarray:
        # result_data用于存储dicom文件中的信息
        result_data = []
        # 读取dicom文件并将信息保存
        for dicom_filename in os.listdir(dicom_dir):
            dicom_path = os.path.join(dicom_dir,dicom_filename)
            dicom_img = pydicom.dcmread(dicom_path)
            result_data.append(dicom_img.pixel_array)

        return np.array(result_data)
    
    def transform_label(self,label:np.array) -> np.array:
        # result_data用于存储转换后的label
        result_data = []
        # 转换label
        for i in range(label.shape[-1]):
            extract_label = label[:,:,i].T #标签需要转置
            result_data.append(extract_label)

        return np.array(result_data)


# test_code

# class Data_statistics
'''
test_path = 'D:/ILD/dicomandnii'
data_statistics = Data_statistics(test_path)
data_statistics.save_statistics_information('./experiment/data_statistics.xlsx')
'''

# class Data_preprocessor
'''
original_img_dir = './ILDAIDATA'
np_data_dir = './2Dimg_data'
data_preprocessor = Data_preprocessor(original_img_dir,np_data_dir)
data_preprocessor.original_dicomfile_rename()
data_preprocessor.data_and_label_generator()
data_preprocessor.data_check_and_show(4)
'''

# class Dataset2D
'''
dataset = Dataset2D('./2Dimg_data',4)
dataloader = DataLoader(dataset,batch_size=2,shuffle=False)

for data,label in dataloader:
    plt.imshow(data[0])
    plt.imshow(label[0],alpha=0.2,cmap='Reds')
    plt.show()
'''

# class Dataset2D_multiply
'''
dataset = Dataset2D_multiply('./2Dimg_data',10)
dataloader = DataLoader(dataset,batch_size=2,shuffle=False)

for data,label in dataloader:
    print(data.shape)
    print(label.shape)
'''

# class Dataset3D_from_fact
'''
dataset = Dataset3D_from_fact('./ILDAIDATA',4)
dataloader = DataLoader(dataset,batch_size=1)

for data_tensor,label_tensor in dataloader:
    data = data_tensor.numpy()
    label = label_tensor.numpy()
    print(data.shape)
    print(label.shape)
    for j in range(label.shape[0]):
        for i in range(label.shape[3]):
            if np.sum(label[j, i , : , :]) != 0:
                plt.imshow(data[j,i,:,:])
                plt.imshow(label[j,i,:,:],alpha=0.2)
                plt.show()
'''
if __name__ == '__main__':
    # 以下为数据处理过程的代码示例

    # 1.统计数据集基本信息，非必要处理，可以跳过
    '''
    # 设置初始化参数，包括Fact软件导出的数据集存放位置(data_dir)、统计信息存放位置(excel_path)
    data_dir需要具有以下结构：
        root
            |_sample1
                |_dicom
                    |_xxx.dcm
                |_result
                    |_labeln.nii
            |_sample2
            ....
            |_samplen

    excel 文件路径可以随意指定，无特殊要求，正常情况下保存在'./experiment'下
    '''
    print('统计样本数据中。。。。。')
    data_dir = '.././ILDAIDATA'
    excel_path = '.././experiment/data_statistics_test.xlsx'
    data_statistics = Data_statistics(data_dir)
    data_statistics.save_statistics_information(excel_path)
    print('样本统计完成！')

    # 2.数据预处理, 主要包含两步骤
    '''
    一、dicom文件的重命名（无论是2D还是3D数据集，从fact直接导出的文件必须进行一次重命名以保证数据和标签一致）
    二、根据dicom和nii文件生成npy文件，用于生成2D数据集，对于3D数据集而言该步骤不是必须的，可以跳过。
    '''
    print('数据预处理中。。。。。。')
    original_img_dir = '.././ILDAIDATA'
    np_data_dir = '.././2Dimg_data'

    data_preprocessor = Data_preprocessor(original_img_dir,np_data_dir)
    data_preprocessor.original_dicomfile_rename()
    data_preprocessor.data_and_label_generator()
    print('数据预处理完成')
    # 3.生成数据集，以生成2D单分类数据集为例
    # 指定数据集路径
    print('准备生成数据集')
    dataset_path = '.././2Dimg_data'
    # 生成dataset,参数label_index用于指定生成那种类别的数据集
    dataset = Dataset2D(dataset_path,label_index=4)
    # 加载数据集
    dataloader = DataLoader(dataset=dataset,batch_size=6,shuffle=True)

    # 获得数据集数据，可以直接送入网络进行训练
    for data,label in dataloader:
        print(f'data:{data.shape}\tlabel:{label.shape}')

