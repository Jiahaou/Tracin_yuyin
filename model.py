# 使用torch.nn包来构建神经网络.
import torch.nn as nn
import torch.nn.functional as F
import torch
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
            attention = F.softmax(torch.matmul(K,Q),dim=2)
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
