import random

import numpy as np
import torch
import torchvision
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt


# from network import Net


class MineNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (13-3)/2+1=6
        )  # 6*6*128=9126

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        # softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.logsoftmax(x)
        return x


# 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

n_epochs = 3  # epoch的数量定义了我们将循环整个训练数据集的次数

# 我们将使用batch_size_train=20进行训练，并使用batch_size_test=20对这个数据集进行测试。
batch_size_train = 20  # 训练批次，即每个批次共有20条数据
batch_size_test = 20  # 测试批次，即每个批次共有20条数据

# learning_rate和momentum是我们稍后将使用的优化器的超参数
learning_rate = 0.01  # 学习率 越小越慢
momentum = 0.9
weight_decay = 5e-4

log_interval = 10  # 每10批次训练打印一次当前训练进度
random_seed = 1  # 固定随机数种子
torch.manual_seed(random_seed)  # 对于可重复的实验，我们必须为任何使用随机数产生的东西设置随机种子

Use_gpu = torch.cuda.is_available()  # GPU 是否可用

# 采用cpu还是gpu进行计算，如果gpu能用就用gpu，否则用cpu
if Use_gpu:
    DEVICE = torch.device('cuda')
    # 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed_all(random_seed)
else:
    DEVICE = torch.device('cpu')

# 数据集根路径 里面有train和test两个子文件夹
data_dir = "../dc_2000"

# 图片处理
image_datasets = {
    "train": datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
        transforms.Resize((256, 256)),  # 先调整图片大小至256x256
        transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化，数值是用ImageNet给出的数值
    ])),
    "test": datasets.ImageFolder(os.path.join(data_dir, "test"), transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]))
}

# 数据加载器
data_loader = {
    "train": torch.utils.data.DataLoader(
        image_datasets["train"],  # 经过处理的图片数据集
        batch_size=batch_size_train,  # 每批次加载的图片数量
        shuffle=True  # 是否打乱数据
    ),
    "test": torch.utils.data.DataLoader(
        image_datasets["test"],
        batch_size=batch_size_test,
        shuffle=True,
    )
}
train_loader = data_loader["train"]
test_loader = data_loader["test"]

# iter 返回一个迭代器，next为获取下一个即第一个元素
X_example, y_example = next(iter(data_loader["train"]))
print("X_example: ", X_example.shape)
print("y_example: ", y_example.shape)

example_classees = image_datasets["train"].classes  # 获取类别
print("example_classees: ", example_classees)

index_classes = image_datasets["train"].class_to_idx  # 获取类别索引
print("index_classes: ", index_classes)

# 我们可以使用matplotlib来绘制其中的一些图片
fig = plt.figure()  # 创建图像
for i in range(6):  # 取出头六条数据
    plt.subplot(2, 3, i + 1)  # 将图像分成 2x3 个子图对6条数据进行绘制，子图编号为1~N，所以i+1
    plt.tight_layout()  # 调整子图之间和周围的填充。
    # imshow 展示图片
    plt.imshow(
        X_example[i][0],  # 表示第i+1个28x28的像素数据
        # cmap='gray',  # cmap='gray'指定该图为灰度图
        interpolation='none'  # 不插值
    )
    plt.title(f"Real Label: {example_classees[y_example[i]]}")  # 设置标题
    plt.xticks([])  # 不显示x轴刻度
    plt.yticks([])  # 不显示y轴刻度
plt.show()  # 展示所有子图

# 迁移学习模型
# model = models.vgg16(pretrained=True)  # 加载预训练模型
# for parma in model.parameters():
#     parma.requires_grad = False  # 屏蔽预训练模型的权重，只训练最后一层的全连接的权重
#
# model.classifier = nn.Sequential(nn.Linear(25088, 4096),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.5),
#                                  nn.Linear(4096, 4096),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.5),
#                                  nn.Linear(4096, 2))
#
# for index, parma in enumerate(model.classifier.parameters()):
#     if index == 6:
#         parma.requires_grad = True
#
# # 损失函数和优化器
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
# scheduler = StepLR(optimizer, step_size=3)
# loss_f = nn.CrossEntropyLoss()


# 自定义模型
model = MineNet()
# model = MyConvNet().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
scheduler = StepLR(optimizer, step_size=5)
loss_f = nn.CrossEntropyLoss()

# 如果使用GPU，将模型放到GPU上
if Use_gpu:
    model = model.cuda()

# 在x轴上，我们希望显示网络在培训期间看到的培训示例的数量。
train_losses = []  # 训练损失记录
train_counter = []

# 我们还创建了两个列表来节省培训和测试损失。
test_losses = []  # 测试损失记录
test_counter = [i * len(data_loader["train"].dataset) for i in range(n_epochs + 1)]


def train(epoch):
    model.train()  # 将网络设置为训练模式
    train_loss = 0.0  # 训练损失
    train_acc = 0.0  # 训练准确率

    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for batch_idx, (data, target) in enumerate(data_loader["train"], 1):  # batch为索引，data为数据，获取从第二个元素开始的数据
        data, target = data.to(DEVICE), target.to(DEVICE)
        # data 为数据集，target为标签
        if Use_gpu:  # 判断是否使用GPU
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        y_pred = model(data)  # 将数据输入网络，得到输出

        _, pred = torch.max(y_pred.data, 1)  # 获取最大值的索引，即预测的类别
        optimizer.zero_grad()  # 清空梯度
        loss = loss_f(y_pred, target)  # 计算损失
        loss.backward()  # 反向传播计算当前梯度# 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
        optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
        train_acc += pred.eq(target.view_as(pred)).sum().item()  # 计算训练准确率
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,  # 当前训练的轮数
                batch_idx * len(data),  # 当前已训练的样本数
                len(train_loader.dataset),  # 训练集总样本数
                100. * batch_idx / len(train_loader),  # 当前训练进度
                loss.item()  # 当前损失
            ))
            train_losses.append(loss.item())  # 每十次记录一次损失
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(data_loader["train"].dataset)))  # 每十次记录一次训练样本数
            # .state_dict() 返回一个包含模块整体状态的字典。
            # 参数和持久性缓冲区（例如，运行平均数）都是包括在内。键是相应的参数和缓冲区名称。设置为 "无 "的参数和缓冲区不包括在内。
            # torch.save(model.state_dict(), './model.pth')  # 每十次保存一次模型
            # torch.save(optimizer.state_dict(), './optimizer.pth')  # 每十次保存一次优化器
    train_loss *= batch_size_train
    train_loss /= len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)
    print('\ntrain epoch: {}\tloss: {:.6f}\taccuracy:{:.4f}% '.format(epoch, train_loss, 100. * train_acc))
    scheduler.step()  # 更新学习率
    return train_loss, train_acc


def test(epoch):
    model.eval()  # 将网络设置为评估模式
    test_loss = 0  # 测试损失
    correct = 0  # 正确预测的样本数

    for data, target in data_loader["test"]:
        # 使用上下文管理器no_grad()，我们可以避免将生成网络输出的计算结果存储在计算图中。
        # 禁用梯度计算的上下文管理器。在评估模型时，这是非常有用的，因为它可以减少内存使用，并加快计算。
        # 禁用梯度计算对于推理很有用，当你确定你不会调用: meth:`Tensor.backward()`。它将减少内存计算的内存消耗
        with torch.no_grad():
            data, target = data.to(DEVICE), target.to(DEVICE)
            y_pred = model(data)  # 将数据输入网络，得到输出
            test_loss += loss_f(y_pred, target).item()  # 计算损失

            pred = torch.max(y_pred.data, 1)[1]  # 获取最大值的索引，即预测的类别
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累加预测正确数

    test_loss *= batch_size_test
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    test_losses.append(test_loss)  # 记录当前平均损失
    print("\nTest set: epoch{} average loss: {:.4f}, accuracy: {}/{} ({:.4f}%) \n"
          .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * test_acc))
    return test_loss, 100. * test_acc


# train(1)
# test()
# for epoch in range(1, n_epochs + 1):
#     train(epoch)  # 训练 epoch 表示当前是第几轮训练
#     test()  # 测试一下当前模型的效果

# fig = plt.figure()  # 创建图像
# plt.plot(train_counter, train_losses, color='blue')  # 画出训练损失曲线
# plt.scatter(test_counter, test_losses, color='red')  # 画出测试损失散点图
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # 图例标识
# plt.xlabel('number of training examples seen')  # x轴标签
# plt.ylabel('negative log likelihood loss')  # y轴标签

model = torchvision.models.vgg16()
# 加载模型参数
model.load_state_dict(torch.load('model.pth'))
# 加载优化器参数
model.load_state_dict(torch.load('optimizer.pth'))


def showDemoDigit():
    examples = enumerate(data_loader["test"])  # 枚举测试集
    batch_idx, (example_data, example_targets) = next(examples)  # 获取第一个批次的数据
    with torch.no_grad():  # 不计算梯度，节省内存
        y_pred = model(example_data)  # 将数据输入网络，得到输出
        _, pred = torch.max(y_pred.data, 1)
    fig = plt.figure()  # 创建图像

    for i in range(6):  # 画出前6个样本的预测结果
        plt.subplot(2, 3, i + 1)  # 2行3列，第 i+1 个子图
        plt.tight_layout()  # 自动适配子图参数，使之填充整个图像区域
        index = random.randint(0, len(example_data) - 1)
        plt.imshow(example_data[index][0], interpolation='none')  # 画出第 i 个样本
        plt.title(
            f"Real Label: {example_classees[example_targets[index]]}\nPredict Label: {example_classees[pred[index]]}")  # 标题为预测结果
        plt.xticks([])  # 不显示 x 轴刻度
        plt.yticks([])  # 不显示 y 轴刻度
    plt.show()  # 显示图像


showDemoDigit()
