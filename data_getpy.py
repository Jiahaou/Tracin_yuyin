import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from feture_extractor import FeatureExtractor

import pickle
import os
import glob
import librosa
from tqdm import tqdm
import numpy as np

class get_data(object):
    def __init__(self, featuresExist,featuresFileName,WAV_PATH,rate,FEATURES_TO_USE,toSaveFeatures,BATCH_SIZE,impro_or_script):
        self.featuresExist = featuresExist
        self.WAV_PATH= WAV_PATH
        self.featuresFileName = featuresFileName
        self.FEATURES_TO_USE=FEATURES_TO_USE
        self.rate = rate
        self.toSaveFeatures =toSaveFeatures
        self.BATCH_SIZE =BATCH_SIZE
        self.impro_or_script=impro_or_script


    def getdata_Tracin(self, path, RATE=16000):  # t=2 按照时间间隔为2秒来进行划分
        #  把每条语句切分成多个2秒的片段，片段之间存在1秒的重叠（测试集是1.6秒重叠）
        path = path.rstrip('/')  # 删除路径后面的'/'符号
        wav_files = glob.glob(path + '/*.wav')  # 获取声音文件

        LABEL_DICT1 = {  # 情绪标签文件
            '01': 'neutral',
            # '02': 'frustration',
            # '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            # '06': 'fearful',
            '07': 'happy',  # excitement->happy
            # '08': 'surprised'
        }
        #  预处理
        n=500

        train_files = []  # 训练集
        valid_files = []  # 测试集
        # 将元组转化为列表list()
        train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))  # 随机获取80%的训练数据
        valid_indices = list(set(range(n)) - set(train_indices))  # 将剩下的数据作为测试数据
        train_indices = list(set(range(n)) - set(valid_indices))
        for i in train_indices:  # 分别将数据放入相应的列表
            train_files.append(wav_files[i])
        for i in valid_indices:
            valid_files.append(wav_files[i])
        train_X = []
        train_y = []
        train_z = []
        print("constructing meta dictionary for {}...".format(path))
        # 这里的enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        for i, wav_file in enumerate(tqdm(train_files)):  # 这里是对于训练数据来处理
            label = str(os.path.basename(wav_file).split('-')[2])  # 函数返回path最后的文件名，获取标签的名称

            if (label not in LABEL_DICT1):  # 如果标签不在标签字典里面，就执行下面的语句，否则跳出if
                continue
            if (self.impro_or_script != 'all' and (self.impro_or_script not in wav_file)):  # 训练数据仅仅是采用了随机发挥的文本
                continue
            label = LABEL_DICT1[label]
            # 将wav切分成2秒的片段，并丢弃少于2秒的部分
            train_data, _ = librosa.load(wav_file, sr=RATE)  # sr=1600000
            train_name = str(os.path.basename(wav_file).split('\\')[0])
            train_y.append(label)
            train_X.append(train_data)
            train_z.append(train_name)

        test_X = []
        test_y = []
        test_z = []
        for i, wav_file in enumerate(tqdm(valid_files)):
            label = str(os.path.basename(wav_file).split('-')[2])
            if (label not in LABEL_DICT1):
                continue
            # if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            #     continue
            label = LABEL_DICT1[label]
            wav_data, _ = librosa.load(wav_file, sr=RATE)
            test_z.append(str(os.path.basename(wav_file).split('\\')[0]))
            test_X.append(wav_data)
            test_y.append(label)



        length = max([len(i) for i in train_X])
        length2 = max([len(i) for i in test_X])
        length_all = max(length2, length)
        for i in range(len(train_X)):
            if len(train_X[i]) <= length_all:  # 使所有的ndarry长度一致
                A = np.zeros(length_all)
                A[:len(train_X[i])]=train_X[i]
                train_X[i]=A
        for i in range(len(test_X)):
            if len(test_X[i]) <= length_all:  # 使所有的ndarry长度一致
                B = np.zeros(length_all)
                B[:len(test_X[i])]=test_X[i]
                test_X[i]=B

        test_X = np.row_stack(test_X)  # 所有行合并
        test_y = np.array(test_y)
        test_z = np.array(test_z)
        assert len(test_X) == len(test_y), "X length and y length must match! X shape: {}, y length: {}".format(
            test_X.shape, test_y.shape)





        train_X = np.row_stack(train_X)
        train_y = np.array(train_y)
        train_z = np.array(train_z)
        assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
            train_X.shape, train_y.shape)
        return train_X, train_y, train_z, test_X, test_y, test_z

    def process_data(self,path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000): # t=2 按照时间间隔为2秒来进行划分
        #  把每条语句切分成多个2秒的片段，片段之间存在1秒的重叠（测试集是1.6秒重叠）
        path = path.rstrip('/')  # 删除路径后面的'/'符号
        wav_files = glob.glob(path + '/*.wav')  # 获取声音文件
        meta_dict = {}
        val_dict = {}
        LABEL_DICT1 = {     #  情绪标签文件
            '01': 'neutral',
            # '02': 'frustration',
            # '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            # '06': 'fearful',
            '07': 'happy',  # excitement->happy
            # '08': 'surprised'
        }
    #  预处理
        n = 120
        wav_files_2 = wav_files[:n]
        train_files = []    #  训练集
        valid_files = []    #  测试集
        # 将元组转化为列表list()
        train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))  # 随机获取80%的训练数据
        valid_indices = list(set(range(n)) - set(train_indices))  #  将剩下的数据作为测试数据
        train_indices = list(set(range(n)) - set(valid_indices))
        for i in train_indices: # 分别将数据放入相应的列表
            train_files.append(wav_files_2[i])
        for i in valid_indices:
            valid_files.append(wav_files_2[i])

        print("constructing meta dictionary for {}...".format(path))
        # 这里的enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        for i, wav_file in enumerate(tqdm(train_files)):  #  这里是对于训练数据来处理
            label = str(os.path.basename(wav_file).split('-')[2]) # 函数返回path最后的文件名，获取标签的名称
            if (label not in LABEL_DICT1): # 如果标签不在标签字典里面，就执行下面的语句，否则跳出if
                continue
            if (self.impro_or_script != 'all' and (self.impro_or_script not in wav_file)): # 训练数据仅仅是采用了随机发挥的文本
                continue
            label = LABEL_DICT1[label]
            # 将wav切分成2秒的片段，并丢弃少于2秒的部分
            wav_data, _ = librosa.load(wav_file, sr=RATE)  # sr=1600000
            X1 = []   # 定义了两个列表
            y1 = []
            z1 = []
            index = 0  # 索引的初值为0
            if (t * RATE >= len(wav_data)):
                continue
            k=0
            while (index + t * RATE < len(wav_data)):
                X1.append(wav_data[int(index):int(index + t * RATE)])
                y1.append(label)

                assert t - train_overlap > 0
                index += int((t - train_overlap) * RATE)
                k+=1
                z1.append(str(os.path.basename(wav_file).split('\\')[0]+str(k)))
            X1 = np.array(X1)
            meta_dict[i] = {
                'X': X1,
                'y': y1,
                'z': z1,
                'path': wav_file
            }

        print("building X, y, z...")
        train_X = []
        train_y = []
        train_z = []
        for k in meta_dict:
            train_X.append(meta_dict[k]['X'])
            train_y += meta_dict[k]['y']
            train_z += meta_dict[k]['z']
        train_X = np.row_stack(train_X)
        train_y = np.array(train_y)
        train_z = np.array(train_z)
        assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
            train_X.shape, train_y.shape)

        if (val_overlap >= t):
            val_overlap = t / 2
        for i, wav_file in enumerate(tqdm(valid_files)):
            label = str(os.path.basename(wav_file).split('-')[2])
            if (label not in LABEL_DICT1):
                continue
            # if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            #     continue
            label = LABEL_DICT1[label]
            wav_data, _ = librosa.load(wav_file, sr=RATE)
            X1 = []
            y1 = []
            z1 = []
            index = 0
            if (t * RATE >= len(wav_data)):
                continue
            while (index + t * RATE < len(wav_data)):
                X1.append(wav_data[int(index):int(index + t * RATE)])
                y1.append(label)
                index += int((t - val_overlap) * RATE)
                z1.append(str(os.path.basename(wav_file).split('\\')[0]))
            X1 = np.array(X1)
            val_dict[i] = {
                'X': X1,
                'y': y1,
                'z': z1,
                'path': wav_file
            }
        print("building X, y...")
        test_X = []
        test_y = []
        test_z = []
        for t in val_dict:
            test_X.append(val_dict[t]['X'])
            test_y += val_dict[t]['y']
            test_z += val_dict[t]['z']
        test_X = np.row_stack(test_X)  # 所有行合并
        test_y = np.array(test_y)
        test_z = np.array(test_z)
        assert len(test_X) == len(test_y), "X length and y length must match! X shape: {}, y length: {}".format(
            test_X.shape, test_y.shape)
        return train_X, train_y,train_z ,test_X, test_y,test_z
