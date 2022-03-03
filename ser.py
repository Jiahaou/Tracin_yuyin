"""引入依赖项"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim      #  进行了优化操作
from torch.utils.data import Dataset,DataLoader  # 批量提取数据，并且输出和标签相对应输出
import numpy as np
import librosa         # 与音频处理相关的库
from tqdm import tqdm  # 使用进度条，方便显示
import glob            #  glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
import os
import pickle          # pickle是二进制序列化格式;
import random          # 随机的概念: 在某个范围内取到的每一个值的概率是相同的
import logging         # 日志文件


"""定义注意力卷积模型"""
class MACNN(nn.Module):    # 定义注意力卷积模型
    # 相当于输入的特征值的不同数量，详见ppt
    def __init__(self, attention_heads=8, attention_hidden=256, out_size=4):
        super(MACNN, self).__init__()  # 继承函数__init__()中的相关的参数
        self.attention_heads = attention_heads # 定义一共有几个注意力头
        self.attention_hidden = attention_hidden # 定义注意力层的层数
        # 接下来就是定义的各个卷积层
        # kernel_size和padding之间的关系：padding=int(kernel_size - 1)//2 （表示向下取整的意思）
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=attention_hidden, padding=1)
        #池化操作
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        # 参数为out_channels 进行数据的归一化处理
        self.bn1a = nn.BatchNorm2d(8)  # 参数为待处理数据通道数
        self.bn1b = nn.BatchNorm2d(8)  # 每一次卷积操作之后都会进行一次数据的标准化操作
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(attention_hidden)
        # 自适应平均池化函数 参数是output_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 是用来设置网络的全连接层的
        self.fc = nn.Linear(in_features=self.attention_hidden, out_features=out_size)
        # 以0.5的概率让神经元置0
        self.dropout = nn.Dropout(0.5)
        # 对于cnn前馈神经网络如果前馈一次写一个forward函数会有些麻烦
        self.attention_query = nn.ModuleList() # 这里添加的是几个层而不是模型
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()
        for i in range(self.attention_heads):
          self.attention_query.append(nn.Linear(90, 90))
          self.attention_key.append(nn.Linear(90, 90))
          self.attention_value.append(nn.Linear(90, 90))

# 定义前向传播函数 就是模型照着来

    def forward(self, *input):
        xa = self.conv1a(input[0])  # 这个input[0]
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        # 将两个矩阵按照列拼接起来
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        height = x.shape[2]
        width = x.shape[3]
        # 重新定义四维张量的形状
        x = x.reshape(x.shape[0], x.shape[1], 1, -1)
        # Head Fusion
        attn = None  # 用于计算self-attention
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x).transpose(2,3)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.matmul(K,Q))
            attention = torch.matmul(V,attention)
            attention=attention.reshape(attention.shape[0],attention.shape[1],height,width)
            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

"""设定随机种子"""
# 在使用pytorch框架搭建模型的时候，模型中的参数都是进行初始化的，
# 且每次初始化的结果不同，这就导致每次的训练模型不一样，要想在程序不变的情况下，
# 使得每次的输出结果一致，那就要设定随机种子。
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

"""一些训练参数"""
setup_seed(111111)  # seed( ) 用于指定随机数生成时所用算法开始的整数值
attention_head = 4
attention_hidden = 32
learning_rate = 0.001  # 学习率设置初值
Epochs = 50    #  总共次数的训练迭代
BATCH_SIZE = 32  # 一次训练所选取的样本数。
FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
impro_or_script = 'impro'
featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
featuresExist = False
toSaveFeatures = True
WAV_PATH = "..\SER\IEMOCAP"  # 声音文件的显示路径
RATE = 16000
MODEL_NAME = 'MACNN'    # 使用上面定义的
MODEL_PATH = 'models/{}_{}_1.pth'.format(MODEL_NAME, FEATURES_TO_USE) # 定义的模型的路径

"""定义预处理模块"""
# 预处理这部分做了下面几件事：
# 1.把数据集以语句为单位，分成80%的训练集和20%的测试集
# 2.把每条语句切分成多个2秒的片段，片段之间存在1秒的重叠（测试集是1.6秒重叠）
# 3.把切分好的片段标上源语句的标签
# 处理数据 clear
def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000): # t=2 按照时间间隔为2秒来进行划分
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
    n = len(wav_files)  # 一共有多少个语音数据
    train_files = []    #  训练集
    test_files = []    #  测试集
    # 将元组转化为列表list()
    train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))  # 随机获取80%的训练数据
    test_indices = list(set(range(n)) - set(train_indices))  #  将剩下的数据作为测试数据
    for i in train_indices: # 分别将数据放入相应的列表
        train_files.append(wav_files[i])
    for i in test_indices:
        test_files.append(wav_files[i])

    print("constructing meta dictionary for {}...".format(path))
    # 这里的enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    for i, wav_file in enumerate(tqdm(train_files)):  #  这里是对于训练数据来处理
        label = str(os.path.basename(wav_file).split('-')[2]) # 函数返回path最后的文件名，获取标签的名称
        if (label not in LABEL_DICT1): # 如果标签不在标签字典里面，就执行下面的语句，否则跳出if
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)): # 训练数据仅仅是采用了随机发挥的文本
            continue
        label = LABEL_DICT1[label]
        # 将wav切分成2秒的片段，并丢弃少于2秒的部分
        wav_data, _ = librosa.load(wav_file, sr=RATE)  # sr=1600000
        X1 = []   # 定义了两个列表
        y1 = []
        index = 0  # 索引的初值为0
        if (t * RATE >= len(wav_data)):
            continue

        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            assert t - train_overlap > 0
            index += int((t - train_overlap) * RATE)
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    train_X = []
    train_y = []
    for k in meta_dict:
        train_X.append(meta_dict[k]['X'])
        train_y += meta_dict[k]['y']
    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
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
        index = 0
        if (t * RATE >= len(wav_data)):
            continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            index += int((t - val_overlap) * RATE)

        X1 = np.array(X1)
        val_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    return train_X, train_y, val_dict

"""定义特征提取模块"""
#特征提取器
class FeatureExtractor(object):
    def __init__(self, rate):
        self.rate = rate
    # 定义函数获取特征提取的方法
    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'pase')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('logfbank'):
            X_features = self.get_logfbank(X)  # 调用下面的定义的方法
        if features_to_use in ('mfcc'):
            X_features = self.get_mfcc(X,26)
        if features_to_use in ('fbank'):
            X_features = self.get_fbank(X)
        if features_to_use in ('melspectrogram'):
            X_features = self.get_melspectrogram(X)
        if features_to_use in ('spectrogram'):
            # spectrogram是一个MATLAB函数，使用短时傅里叶变换得到信号的频谱图。当使用时无输出参数，会自动绘制频谱图；有输出参数，则会返回输入信号的短时傅里叶变换。
            X_features = self.get_spectrogram(X)
        if features_to_use in ('pase'):
            X_features = self.get_Pase(X)
        return X_features

    def get_logfbank(self, X):  # 用不同的方法来获取特征处理结果
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features
# 提取特征 用librosa提取mfcc特征
    def get_mfcc(self, X, n_mfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(y=x, sr=self.rate, n_mfcc=n_mfcc)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.rate)
            mel = np.log10(mel + 1e-10)
            return mel

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features


    def get_Pase(self,X):
        return X

# pytorch中用于读取数据集的类
class DataSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        # 这个方法返回与指定键想关联的值。对序列来说，键应该是0~n-1的整数，其中n为序列的长度。对映射来说，键可以是任何类型。
        x = self.X[index]
        x = torch.from_numpy(x)
        x = x.float()
        y = self.Y[index]
        y = dict[y]
        y = y.long()
        return x, y

    def __len__(self):
        return len(self.X)

"""训练模型并测试"""
if __name__ == '__main__':
    train_data = DataSet(train_X_features, train_y)   # 初始化了X和Y的值
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    model = MACNN(attention_head, attention_hidden)  # 调用模型
    if torch.cuda.is_available():  # 使用GPU
      model = model.cuda()
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6) # 更新参数优化
    maxWA = 0
    maxUA = 0
    maxACC = 0
    # 一个epoch的训练过程
    for epoch in range(Epochs):
        model.train()
        print_loss = 0
        for _, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x.unsqueeze(1))
            loss = criterion(out, y.squeeze(1))
            print_loss += loss.data.item() * BATCH_SIZE
            optimizer.zero_grad()   # 反向求梯度
            loss.backward()         #  反向求损失函数
            optimizer.step()        #  跟新参数
        print('epoch: {}, loss: {:.4}'.format(epoch, print_loss / len(train_X_features))) # 平均损失

        if (epoch > 0 and epoch % 10 == 0):
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # 测试
        model.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)
        #验证
        for _, i in enumerate(valid_features_dict):
            x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
            x = torch.from_numpy(x).float()
            y = dict[y[0]].long()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            if (x.size(0) == 1):
                x = torch.cat((x, x), 0)
            out = model(x.unsqueeze(1))
            # out = model(x)
            pred = torch.Tensor([0, 0, 0, 0])
            if torch.cuda.is_available():
              pred=pred.cuda()
            for j in range(out.size(0)):
                pred += out[j]
            pred = pred / out.size(0)
            pred = torch.max(pred, 0)[1]
            if (pred == y):
                num_correct += 1
            matrix[int(y), int(pred)] += 1
    # 在验证集上计算正确率
        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i, j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct / len(valid_features_dict)
        if (maxWA < WA):
            maxWA = WA
        if (maxUA < sum(UA) / 4):
            maxUA = sum(UA) / 4
            # 当综合准确率提升时保存模型
        if (maxACC < (WA + sum(UA) / 4)):
            maxACC = WA + sum(UA) / 4
            torch.save(model.state_dict(), MODEL_PATH)
            print('saving model,epoch:{},WA:{},UA:{}'.format(epoch, WA, sum(UA) / 4))
        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))

        print(matrix)