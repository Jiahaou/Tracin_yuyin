import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle          # pickle是二进制序列化格式;
import random          # 随机的概念: 在某个范围内取到的每一个值的概率是相同的
import logging
import time
import os
from ser import DataSet,DataLoader,MACNN,FeatureExtractor,setup_seed,process_data
from torch.autograd import grad
from pif.influence_functions_new import pick_gradient,param_vec_dot_product
from pif.influence_functions_new import calc_all_grad

"""一些训练参数"""
setup_seed(111111)  # seed( ) 用于指定随机数生成时所用算法开始的整数值
attention_head=4
attention_hidden=32
learning_rate=0.001
Epochs=50
BATCH_SIZE=32
FEATURES_TO_USE='mfcc'
impro_or_script='impro'
featuresFileName='features_{}_{}.pkl'.format(FEATURES_TO_USE,impro_or_script)
featuresExist=False
toSaveFeatures=True
RATE=16000
MODEL_NAME='MACNN'
# MODEL_PATH='models/{}_{}_1.pth'.format(MODEL_NAME,FEATURES_TO_USE)
MODEL_PATH='Lenet.pth'
WAV_PATH="..\SER\IEMOCAP/"
if (featuresExist == True):
    with open(featuresFileName, 'rb')as f:
        features = pickle.load(f)
    train_X_features = features['train_X']
    train_y = features['train_y']
    test_X_features = features['test_X']
    test_y = features['test_y']
else:
    logging.info("creating meta dict...")
    train_X, train_y, test_X,test_y = process_data(WAV_PATH, t=2, train_overlap=1)
    print(train_X.shape)
    print(test_X.shape)
    print("getting features")
    logging.info('getting features')
    feature_extractor = FeatureExtractor(rate=RATE)
    train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
    test_X_features = feature_extractor.get_features(FEATURES_TO_USE, test_X)

    # for _, i in enumerate(tqdm(val_dict)):
    #     X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
    #     valid_features_dict[i] = {
    #         'X': X1,
    #         'y': val_dict[i]['y']
    #     }
    if (toSaveFeatures == True):
        features = {'train_X': train_X_features, 'train_y': train_y,
                    'test_X': test_X_features, 'test_y': test_y}
        with open(featuresFileName, 'wb') as f:
            pickle.dump(features, f)

"""用于将对应的情绪标签转化为张量"""
dict = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
    'boring': torch.Tensor([4]),
    'fear': torch.Tensor([5]),
}
train_data = DataSet(train_X_features, train_y)   # 初始化了X和Y的值
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataSet(test_X_features,test_y)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
# model = MACNN(attention_head, attention_hidden)  # 调用模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(    #transformer.compose 执行列表里面的操作
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#normalize数据标准化，第一个数据是mean各通道的均值，第二个数据是std各通道的标准差
#rgb单个通道是0-255，但是imagenet会除以256，或者。totensor（）归一化到0-1之间
#tottensor之后接normalize是因为先归一化，在按通道减去均值，除以方差。使数据分布均匀，更具泛化能力
# train_set = torchvision.datasets.CIFAR10(root='datasets', train=True,
#                                          download=True, transform=transform)
# # 加载训练集，实际过程需要分批次（batch）训练
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
#                                            shuffle=True, num_workers=0)
# # 10000条语音
# test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
#                                         download=False, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
#                                           shuffle=False, num_workers=0)
# 获取测试集中的图像和标签，用于accuracy计算
# test_data_iter = iter(test_loader)
# test_video, test_label = test_data_iter.next()
# net = LeNet()  # 定义训练的网络模型
net = MACNN(attention_head, attention_hidden)
net.to(device)
model_weight_path = "Lenet.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
# 定义优化器（训练参数，学习率）
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
for epoch in range(10):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()
    for _, batch_i in enumerate(tqdm(test_loader)):  # 遍历训练集，step从0开始计算
        inputs = batch_i  # 获取训练集的语音和标签
        labels=train_y
        # labels2=labels.clone()
        # labels2[0] = 9
        # forward + backward + optimize
        outputs = net(inputs.to(device))  # 正向传播
        loss = loss_function(outputs, labels.to(device))  # 计算损失
        grad_z_test = grad(loss, net.parameters())
        grad_z_test = pick_gradient(grad_z_test, net)
        for j, batch_j in enumerate(tqdm(train_loader)):
            inputs_train, labels_train= batch_j  # 获取训练集的语音和标签

            # labels2=labels.clone()
            # labels2[0] = 9
            # forward + backward + optimize
            outputs_train = net(inputs_train.to(device))  # 正向传播
            loss_train = loss_function(outputs_train, labels_train.to(device))  # 计算损失
            grad_z_train = grad(loss_train, net.parameters())
            grad_z_train = pick_gradient(grad_z_train, net)
            score = param_vec_dot_product(grad_z_test, grad_z_train)
            # if t_idx not in train_influences:    #加入json文件保存
            #     train_influences[t_idx] = {'train_dat': (td),
            #                               'if': float(score)}




print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)