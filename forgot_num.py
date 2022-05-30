"""引入依赖项"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim      #  进行了优化操作
from torch.utils.data import Dataset,DataLoader  # 批量提取数据，并且输出和标签相对应输出
import numpy as np
import librosa         # 与音频处理相关的库
from tqdm import tqdm  # 使用进度条，方便显示
          #  glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
import os
import pickle          # pickle是二进制序列化格式;
import random          # 随机的概念: 在某个范围内取到的每一个值的概率是相同的
import logging         # 日志文件
from model import MACNN
from data_getpy import get_data
from feture_extractor import FeatureExtractor
from torch.autograd import grad

from pathlib import Path
from torch.autograd import Variable
import time
from pif.influence_functions_new import pick_gradient,param_vec_dot_product
from pif.utils import save_json

from keras.models import load_model


epoch_acc=[]
def forgot_num(seed,):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    """定义注意力卷积模型"""
    setup_seed(seed)  # seed( ) 用于指定随机数生成时所用算法开始的整数值
    attention_head = 4
    attention_hidden = 32
    learning_rate = 0.001  # 学习率设置初值
    Epochs = 50    #  总共次数的训练迭代
    BATCH_SIZE = 1 # 一次训练所选取的样本数。
    FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
    impro_or_script = 'impro'
    featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
    featuresExist = False
    toSaveFeatures = True
    WAV_PATH = "IEMOCAP/"  # 声音文件的显示路径
    RATE = 16000
    MODEL_NAME = 'MACNN'    # 使用上面定义的
    MODEL_PATH = '{}_{}_222222.pth'.format(MODEL_NAME, FEATURES_TO_USE) # 定义的模型的路径
    dict = {
        'neutral': torch.Tensor([0]),
        'happy': torch.Tensor([1]),
        'sad': torch.Tensor([2]),
        'angry': torch.Tensor([3]),
        'boring': torch.Tensor([4]),
        'fear': torch.Tensor([5]),
    }

    class DataSet(object):
        def __init__(self, X, Y,Z):
            self.X = X
            self.Y = Y
            self.Z = Z

        def __getitem__(self, index):
            # 这个方法返回与指定键想关联的值。对序列来说，键应该是0~n-1的整数，其中n为序列的长度。对映射来说，键可以是任何类型。
            x = self.X[index]
            x = torch.from_numpy(x)
            x = x.float()
            y = self.Y[index]
            y = dict[y]
            y = y.long()
            z = self.Z[index]


            return x, y,z

        def __len__(self):
            return len(self.X)
    topk=1
    getdata = get_data(featuresExist, featuresFileName, WAV_PATH, RATE,
                       FEATURES_TO_USE, toSaveFeatures, BATCH_SIZE, impro_or_script,topk)
    train_X, train_y, train_z, test_X, test_y, test_z = getdata.getdata_Tracin(WAV_PATH)

    feature_extractor = FeatureExtractor(rate=RATE)
    train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
    test_X_features = feature_extractor.get_features(FEATURES_TO_USE, test_X)
    valid_features_dict = {}
    if (toSaveFeatures == True):
        features = {'train_X': train_X_features, 'train_y': train_y, 'train_z': train_z,
                    'test_X': test_X_features, 'test_y': test_y, 'test_z': test_z, }
        with open(featuresFileName, 'wb') as f:
            pickle.dump(features, f)
    train_data = DataSet(train_X_features, train_y, train_z)  # 初始化了X和Y的值
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataSet(test_X_features, test_y, test_z)  # 初始化了X和Y的值
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    model = MACNN(topk,attention_head, attention_hidden)  # 调用模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():  # 使用GPU
        model = model.cuda()
    loss_function = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)  # 更新参数优化
        # 一个epoch的训练过程
    # save_pth="pth/fenlei/retrain_MACNN"#分类
    model.load_state_dict(torch.load("pth/epoch4/retrain_MACNN4.pth"))
    # if "retrain_MACNN.pth":
    #         model.load_state_dict(torch.load("retrain_MACNN.pth"))
    best_acc_epoch1 = 0.0
    maxWA = 0
    maxUA = 0
    maxACC = 0

    model.eval()
    acc_list=[]
    acc = 0.0
    evalidate = []
    with torch.no_grad():  # 全部不求导
        for val_data in test_loader:
            test_inputs, test_labels , _ = val_data
            if test_labels.shape[0] != BATCH_SIZE:
                break
            test_inputs = Variable(torch.unsqueeze(test_inputs, dim=1).float(), requires_grad=False)
            outputs = model(test_inputs.to(device))
            test_labels_1 = test_labels.view(1)
            # loss = loss_function(outputs, test_labels)
            evalidate_y = torch.max(outputs, dim=1)[1]  # 行最大值，返回index
            evalidate.append(int(evalidate_y[0]))
            a = (torch.eq(evalidate_y, test_labels_1.to(device)))
            for i in range(len(a)):
                if (a[i] == True):
                    acc_list.append(1)
                else:
                    acc_list.append(0)
        epoch_acc.append(acc_list)
forgot_num(2222220)
forgot_num(1)
forgot_num(11)
forgot_num(111)
forgot_num(1111)
forgot_num(11111)
forgot_num(111111)
forgot_num(1111111)
forgot_num(2222222)
forgot_num(22222222)
forget_epochnum = 8
acc_list = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(epoch_acc)):
    acc_list[0].append(epoch_acc[0][i])
    acc_list[1].append(epoch_acc[1][i])
    acc_list[2].append(epoch_acc[2][i])
    acc_list[3].append(epoch_acc[3][i])
    acc_list[4].append(epoch_acc[4][i])
    acc_list[5].append(epoch_acc[5][i])
    acc_list[6].append(epoch_acc[6][i])
    acc_list[7].append(epoch_acc[7][i])
    acc_list[8].append(epoch_acc[8][i])
    acc_list[9].append(epoch_acc[9][i])


'''
把不容易忘记的那些数据可以删去，跟上一个epoch比，
若上个为1，这次为1，则score加2，若上次为1，这次为0，则score加1
若上个为0，这次为0，则score加0，若上次为0，这次为1，则score加1
'''

#计算得分值，返回一个list
def get_score(acc_list,forget_epochnum):
    score_list = []
    for i in range(1, forget_epochnum):
        for j in range(len(acc_list[i])):
            if (i == 1):
                if (acc_list[i][j] + acc_list[i - 1][j] == 1):
                    score_list.append(1)
                elif (acc_list[i][j] + acc_list[i - 1][j] == 2):
                    score_list.append(2)
                else:
                    score_list.append(0)
            else:
                if (acc_list[i][j] + acc_list[i - 1][j] == 1):
                    score_list[j] += 1
                elif (acc_list[i][j] + acc_list[i - 1][j] == 2):
                    score_list[j] += 2
    return score_list

#获取难以忘记数据的对应编号
#score_list为得分list，get_percent取出难以忘记数据占总数量的百分比
def get_unforget_batchnum(score_list,get_percent,batchsize):
    score_batch_list = []
    for i in range(int(len(score_list) / batchsize)):
        score_sum = sum(score_list[i * 10:i * 10 + 10])
        score_batch_list.append(score_sum)

    get_num = get_percent * len(score_batch_list)  #取出的数量
    index = int(len(score_batch_list) - get_num)   #排序从小到大，对应的索引
    unforget_num = []
    score_batch_copy = []
    for i in range(len(score_batch_list)):
        score_batch_copy.append(score_batch_list[i])
    score_batch_copy.sort()
    for i in range(len(score_batch_list)):
        if(score_batch_list[i] >= score_batch_copy[index]):
            unforget_num.append(i)
            if(get_num == len(unforget_num)):
                return unforget_num


score_list = get_score(acc_list,forget_epochnum)
unforget_num = get_unforget_batchnum(score_list,0.5,10)
print(unforget_num)
print(1)


