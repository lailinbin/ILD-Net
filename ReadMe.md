# Deep Learning for Auxiliary Diagnostic System of lInterstitial Lung Disease------ILDNet

[TOC]

## Background

## Install
### 1.Create virtual environment by Anaconda
```shell
conda create -n env_name python==3.6
```
### 2.Activate conda environment
```shell
conda activate env_name
```
### 3.Install related libraries
```shell
pip install -r requirements.txt
```
## Usage
### Step1.Data statistics
This step is for dataset statistics information calculation and save. For analysing your own dataset, you can use the following statement:
```shell
python .\code\data_statistics.py -i [dataset_path] -o [excel_path]
```
**Note: the folder of dataset must have the following structure**
```python
ROOT
  |__sample1
    |__dicom
    |__result
  |__sample2
   .......
  |__samplen
```
You can view [Dataset Information](##Dataset Information) for detail of our dataset.
### Step2.Data preprocess
Type following statements on shell to preprocess the dataset.
```shell
python .\code\data_preprocessor.py -origina_img_dir [path of original dataset] -np_data_dir [path of npy data]
```
**Note:the folder of npy data must have the following structure**
```python
ROOT
  |__data
  |__label
```

## Dataset Information
## Maintainers
| Name | E-mail |Address|
| :----: | :----: | :----:|
|Linbin Lai|lailinbin@mail.nwpu.edu.cn| 127 West Youyi Road, Beilin District, Xi'an Shaanxi, 710072, P.R.China.|
