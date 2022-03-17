import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import ser
from pif.influence_functions_new import calc_all_grad


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='datasets', train=True,
                                         download=True, transform=transform)
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                           shuffle=True, num_workers=0)

# 10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                        download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                          shuffle=False, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()
#
# def imshow(img):  # 展示测试集图片和标签
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # print labels
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_label))




net = LeNet()  # 定义训练的网络模型
net.to(device)
# weight_params = []
# for pname, p in net.named_parameters():
#     if ( 'fc' in pname and 'weight' in pname):  #'conv' or
#         weight_params += [p]


loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
# 定义优化器（训练参数，学习率）
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.Adam([
#             {'params': weight_params}],
#             lr=0.001,
#             )

for epoch in range(10):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        inputs, labels = data  # 获取训练集的图像和标签
        optimizer.zero_grad()  # 清除历史梯度


        # labels2=labels.clone()
        # labels2[0] = 9
        # forward + backward + optimize
        outputs = net(inputs.to(device))  # 正向传播
        loss = loss_function(outputs, labels.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        if step % 1000 == 999:  # print every 1000 mini-batches，每1000步打印一次
            with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image.to(device))  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                      (epoch + 1, step + 1, running_loss / 500, accuracy))

                print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
                running_loss = 0.0

print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)