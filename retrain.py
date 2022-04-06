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




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
"""定义注意力卷积模型"""
setup_seed(2222220)  # seed( ) 用于指定随机数生成时所用算法开始的整数值
attention_head = 4
attention_hidden = 32
learning_rate = 0.001  # 学习率设置初值
Epochs = 50    #  总共次数的训练迭代
BATCH_SIZE = 32 # 一次训练所选取的样本数。
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
    'frustration': torch.Tensor([1]),
    'happy': torch.Tensor([2]),
    'sad': torch.Tensor([3]),
    'angry': torch.Tensor([4]),
    'fearful': torch.Tensor([5]),
    'disgust': torch.Tensor([6]),
    'surprised': torch.Tensor([7]),
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
train_X, train_y, train_z, test_X, test_y, test_z = getdata.getdata_retrain(WAV_PATH)
print(train_X.shape)

print("getting features")
logging.info('getting features')
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
    # model_weight_path = "MACNN_mfcc_1.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # model=torch.load(model_weight_path)

    # if "retrain_MACNN.pth":
    #     model.load_state_dict(torch.load("retrain_MACNN.pth"))

if torch.cuda.is_available():  # 使用GPU
    model = model.cuda()
loss_function = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)  # 更新参数优化
    # 一个epoch的训练过程
# train_data = DataSet(train_X_features, train_y,train_z)   # 初始化了X和Y的值
# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_data = DataSet(test_X_features, test_y,test_z)  # 初始化了X和Y的值
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
# model = MACNN(attention_head, attention_hidden)  # 调用模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model_weight_path = "MACNN_mfcc_1.pth"
# # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# # model=torch.load(model_weight_path)
# if torch.cuda.is_available():  # 使用GPU
#   model = model.cuda()
# loss_function = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6) # 更新参数优化
# outdir = Path("result_Tracin")
# # 一个epoch的训练过程
# print(device)
#   # 一个epoch即对整个训练集进行一次训练
#
# all_loss = 0.0
# time_start = time.perf_counter()
# influence_results={}
# # for i, batch_i in enumerate(tqdm(test_loader)):  # 遍历训练集，step从0开始计算
#     # inputs,labels = batch_i  # 获取训练集的语音和标签

epochs=50

save_pth="retrain_MACNN"
# if "retrain_MACNN.pth":
#         model.load_state_dict(torch.load("retrain_MACNN.pth"))
best_acc_epoch1 = 0.0
maxWA = 0
maxUA = 0
maxACC = 0
for epoch in (range(epochs)):
    all_loss = 0.0
    if torch.cuda.is_available():  # 使用GPU
        model = model.cuda()
    model.train()

    time_start = time.perf_counter()
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels,_ = data
        if labels.shape[0]!=BATCH_SIZE:
            break
        inputs = Variable(torch.unsqueeze(inputs, dim=1).float(), requires_grad=False)
        logits = model(inputs.to(device))
        labels = labels.view(32)
        loss = loss_function(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 计算梯度之后更新参数
        # print statistics
        all_loss += loss.item()* BATCH_SIZE  # 计算平均los

    if (epoch > 0 and epoch % 10 == 0):
        learning_rate = learning_rate / 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
   # 测试
    evalidate = []

    model.eval()

    acc = 0.0

    with torch.no_grad():  # 全部不求导
        for val_data in test_loader:
            test_inputs, test_labels , _ = val_data
            if test_labels.shape[0] != BATCH_SIZE:
                break
            test_inputs = Variable(torch.unsqueeze(test_inputs, dim=1).float(), requires_grad=False)
            outputs = model(test_inputs.to(device))
            test_labels_1 = test_labels.view(32)
            # loss = loss_function(outputs, test_labels)
            evalidate_y = torch.max(outputs, dim=1)[1]  # 行最大值，返回index
            evalidate.append(int(evalidate_y[0]))
            acc += torch.eq(evalidate_y, test_labels_1.to(device)).sum().item()

        val_accurate = acc / (len(test_loader)-1)/32
        print('[%d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
              (epoch + 1, all_loss /(len(train_loader)-1)/32, val_accurate))

        print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
        if val_accurate > best_acc_epoch1:
            best_acc_epoch1 = val_accurate
            torch.save(model.state_dict(), save_pth + '.pth')
            print("*****************fresh the data**************")
#         # 测试
#     model.eval()
#     UA = [0, 0, 0, 0]
#     num_correct = 0
#     class_total = [0, 0, 0, 0]
#     matrix = np.mat(np.zeros((4, 4)), dtype=int)
#     #验证
#     for _,test_data in enumerate(test_loader):
#         x, y ,_= test_data
#         # x = torch.from_numpy(x).float()
#         # y = dict[y[0]].long()
#         if torch.cuda.is_available():
#             x = x.cuda()
#             y = y.cuda()
#         if (x.size(0) == 1):
#             x = torch.cat((x, x), 0)
#         x=x.unsqueeze(1)
#
#         out = model(x)
#
#         # out = model(x)
#         pred = torch.Tensor([0, 0, 0, 0])
#         if torch.cuda.is_available():
#           pred=pred.cuda()
#         for j in range(out.size(0)):
#             pred += out[j]
#         pred = pred / out.size(0)
#         pred = torch.max(pred, 0)[1]
#         if (pred == y):
#             num_correct += 1
#         matrix[int(y), int(pred)] += 1
# # 在验证集上计算正确率
#     for i in range(4):
#         for j in range(4):
#             class_total[i] += matrix[i, j]
#         UA[i] = round(matrix[i, i] / class_total[i], 3)
#     WA = num_correct / (len(test_loader)+1)
#     if (maxWA < WA):
#         maxWA = WA
#     if (maxUA < sum(UA) / 4):
#         maxUA = sum(UA) / 4
#         # 当综合准确率提升时保存模型
#     if (maxACC < (WA + sum(UA) / 4)):
#         maxACC = WA + sum(UA) / 4
#         torch.save(model.state_dict(), MODEL_PATH)
#         print('saving model,epoch:{},WA:{},UA:{}'.format(epoch, WA, sum(UA) / 4))
#     print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
#



print('Finished Training'+str(best_acc_epoch1))
