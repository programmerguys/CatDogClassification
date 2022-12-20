import numpy as np  # 线性代数
import pandas as pd  # 数据处理、读取CSV文件
import os  # 获取文件路径
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import matplotlib.pyplot as plt  # 可视化库
import torchvision  # PyTorch视觉模块
from torch.utils.data import Dataset, DataLoader, ConcatDataset  # PyTorch数据加载器
from torchvision import transforms, models  # PyTorch数据转换、预训练模型
from torch.optim.lr_scheduler import *  # PyTorch学习率调度器
import random  # 随机数生成模块
from PIL import Image  # 图像处理库

labelList = ['cat', 'dog']
BATCH_SIZE = 20  # 每批次的大小
EPOCHS = 5  # 迭代次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用cpu或者gpu

cPath = os.getcwd()  # 获取当前路径
train_dir = cPath + '/data/train'  # 训练集路径
test_dir = cPath + '/data/test'  # 测试集路径
train_files = os.listdir(train_dir)  # 训练集文件名下的所有文件
test_files = os.listdir(test_dir)  # 测试集文件名下的所有文件


class CatDogDataset(Dataset):  # 定义猫狗数据集
    def __init__(self, file_list, dir, mode='train', transform=None):  # 初始化
        self.file_list = file_list  # 文件列表
        self.dir = dir  # 文件路径
        self.mode = mode  # 训练或测试模式
        self.transform = transform  # 数据转换
        if self.mode == 'train':  # 如果是训练模式
            self.label = 1 if 'dog' in self.file_list[0] else 0

    def __len__(self):  # 获取数据集长度
        return len(self.file_list)

    def __getitem__(self, idx):  # 获取数据集中的第idx个数据
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))  # 读取图像
        if self.transform:  # 如果有数据转换
            img = self.transform(img)  # 应用数据转换
        if self.mode == 'train':  # 如果是训练模式
            img = img.numpy()  # 将图像转换为numpy数组
            return img.astype('float32'), self.label  # 返回转换后的数组和标签
        else:  # 如果是测试模式
            img = img.numpy()  # 将图像转换为numpy数组
            return img.astype('float32'), self.file_list[idx]  # 返回转换后的数组和文件名


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整图片大小至256x256
    transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化，数值是用ImageNet给出的数值
])

# 把猫的图片和狗的图片分开
cat_files = [tf for tf in train_files if 'cat' in tf]
dog_files = [tf for tf in train_files if 'dog' in tf]

cats = CatDogDataset(cat_files, train_dir, transform=train_transform)  # 猫的数据集类
dogs = CatDogDataset(dog_files, train_dir, transform=train_transform)  # 狗的数据集类

train_set = ConcatDataset([cats, dogs])  # 把猫和狗的数据集合并
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 训练集数据加载器

# 定义测试数据转换
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像的大小调整为(224,224)
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化图像张量
])

test_set = CatDogDataset(test_files, test_dir, mode='test', transform=test_transform)  # 创建测试数据集
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 创建测试数据加载器


class MineNet(nn.Module):  # 定义MineNet类
    def __init__(self, num_classes=2):  # 初始化
        super().__init__()  # 调用父类的初始化函数
        # 定义卷积层和池化层的序列
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

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 定义平均池化层
        # 定义全连接层序列
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 丢弃部分数据
            nn.Linear(128 * 6 * 6, 2048),  # 全连接层，输出维度为2048
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(),  # 丢弃部分数据
            nn.Linear(2048, 512),  # 全连接层，输出维度为512
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(512, num_classes),  # 全连接层，输出维度为num_classes
        )
        # softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)  # 定义LogSoftmax层，按照给定的维度dim计算log softmax

    def forward(self, x):  # 定义前向传播函数
        x = self.features(x)  # 输入经过卷积和池化层序列
        x = self.avgpool(x)  # 输入经过平均池化层
        x = x.view(x.size(0), -1)  # 将多维张量展平成一维
        x = self.classifier(x)  # 输入经过全连接层序列
        x = self.logsoftmax(x)  # 输入经过LogSoftmax层
        return x  # 返回计算结果


def refreshdataloader():
    # 从训练文件列表中提取猫的文件和狗的文件
    cat_files = [tf for tf in train_files if 'cat' in tf]
    dog_files = [tf for tf in train_files if 'dog' in tf]

    # 初始化用于验证的猫的文件列表和狗的文件列表
    val_cat_files = []
    val_dog_files = []

    # 从猫的文件列表和狗的文件列表中随机抽取1250个文件到验证集中
    for _ in range(1250):
        r = random.randint(0, len(cat_files) - 1)  # 随机生成一个索引
        val_cat_files.append(cat_files[r])
        val_dog_files.append(dog_files[r])
        cat_files.remove(cat_files[r])  # 从猫的文件列表中移除抽取出来的文件
        dog_files.remove(dog_files[r])  # 从狗的文件列表中移除抽取出来的文件

    # 获取训练集的dataloader
    train_loader = _extracted_from_refreshdataloader_15(
        cat_files, train_transform, dog_files
    )
    # 获取验证集的dataloader
    val_loader = _extracted_from_refreshdataloader_15(
        val_cat_files, test_transform, val_dog_files
    )
    return train_loader, val_loader


def _extracted_from_refreshdataloader_15(cat_files, transform, dog_files):
    # 创建猫的数据集
    cats = CatDogDataset(cat_files, train_dir, transform=transform)
    # 创建狗的数据集
    dogs = CatDogDataset(dog_files, train_dir, transform=transform)

    # 将猫的数据集和狗的数据集拼接在一起
    train_set = ConcatDataset([cats, dogs])
    # 返回训练集的dataloader
    return DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )


def train(model, device, train_loader, optimizer, epoch, criterion):
    # 将模型设置为训练模式，这会启用训练模式下的 BatchNormalization 和 Dropout 等层
    model.train()
    # 初始化训练损失和准确率
    train_loss = 0.0
    train_acc = 0.0
    # 设置每 10 个批次输出一次信息的参数
    percent = 10

    # 遍历数据加载器中的每一批次数据
    for batch_idx, (sample, target) in enumerate(train_loader):
        # 将输入和目标标签转移到指定的设备上（CPU 或 GPU）
        sample, target = sample.to(device), target.to(device)
        # 将优化器的梯度清零
        optimizer.zero_grad()
        # 将输入传递给模型，获得输出
        output = model(sample)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播梯度
        loss.backward()
        # 更新优化器的权重
        optimizer.step()
        # 将损失从 Tensor 转换为 Python 数值
        loss = loss.item()
        # 累加训练损失
        train_loss += loss
        # 获得分类预测的索引（argmax）
        pred = output.max(1, keepdim=True)[1]
        # 累加准确率
        train_acc += pred.eq(target.view_as(pred)).sum().item()

        # 如果当前批次是第 10 个批次，则输出训练信息
        if (batch_idx + 1) % percent == 0:
            # 计算当前批次处理了多少个样本
            processed_samples = (batch_idx + 1) * len(sample)
            # 计算训练集中有多少个样本
            total_samples = len(train_loader.dataset)
            # 计算当前进度（百分比）
            progress = 100. * batch_idx / len(train_loader)
            # 输出训练信息
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\t'.format(epoch, processed_samples, total_samples,
                                                                             progress, loss))

    # 计算平均训练损失
    train_loss *= BATCH_SIZE
    train_loss /= len(train_loader.dataset)
    # 计算训练准确率
    train_acc = train_acc / len(train_loader.dataset)
    # 输出最终训练信息
    print('\ntrain epoch: {}\tloss: {:.6f}\taccuracy:{:.4f}% '.format(epoch, train_loss, 100. * train_acc))
    # 根据训练准确率调整训练策略
    scheduler.step()

    # 返回训练损失和准确率
    return train_loss, train_acc


def val(model, device, val_loader, epoch, criterion):
    # 设置模型为评估模式
    model.eval()

    # 初始化验证损失和正确样本数
    val_loss = 0.0
    correct = 0

    # 遍历验证数据集
    for sample, target in val_loader:
        # 禁用自动求导
        with torch.no_grad():
            # 将数据转移到设备上
            sample, target = sample.to(device), target.to(device)
            # 进行模型预测
            output = model(sample)

            # 计算验证损失
            val_loss += criterion(output, target).item()
            # 获取每个样本的预测结果
            pred = output.max(1, keepdim=True)[1]
            # 统计正确样本数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均验证损失
    val_loss *= BATCH_SIZE
    val_loss /= len(val_loader.dataset)
    # 计算验证集准确率
    val_acc = correct / len(val_loader.dataset)
    # 打印验证结果
    print("\nval set: epoch{} average loss: {:.4f}, accuracy: {}/{} ({:.4f}%) \n"
          .format(epoch, val_loss, correct, len(val_loader.dataset), 100. * val_acc))
    # 返回验证损失和准确率
    return val_loss, 100. * val_acc


def test(model, device, test_loader, epoch):
    # 设置模型为评估模式
    model.eval()

    # 初始化文件名和预测结果列表
    filename_list = []
    pred_list = []

    # 遍历测试数据集
    for sample, filename in test_loader:
        # 禁用自动求导
        with torch.no_grad():
            # 将数据转移到设备上
            sample = sample.to(device)
            # 进行模型预测
            output = model(sample)
            # 获取每个样本的预测结果
            pred = torch.argmax(output, dim=1)

            # 将文件名和预测结果添加到列表中
            filename_list += [n[:-4] for n in filename]
            pred_list += [p.item() for p in pred]

    # 打印测试结果
    print(f"\ntest epoch: {epoch}\n")

    # 创建提交文件
    submission = pd.DataFrame({"id": filename_list, "label": pred_list})
    submission.to_csv(f'preds_{str(epoch)}.csv', index=False)


def showDemo():
    examples = enumerate(test_loader)  # 枚举测试集
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
            f"Predict Label: {labelList[pred[index]]}")  # 标题为预测结果
        plt.xticks([])  # 不显示 x 轴刻度
        plt.yticks([])  # 不显示 y 轴刻度
    plt.show()  # 显示图像


if __name__ == '__main__':
    samples, labels = iter(train_loader).next()  # 从训练数据加载器中获取一个批次的数据
    plt.figure(figsize=(16, 24))  # 创建绘图窗口
    grid_imgs = torchvision.utils.make_grid(samples[:BATCH_SIZE])  # 将多张图像合并成一张图像
    np_grid_imgs = grid_imgs.numpy()  # 将合并后的图像转换为numpy数组
    # 在tensor中，图像是(batch, width, height)，所以在numpy中必须将其转置为(width, height, batch)才能显示。
    plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))  # 显示合并后的图像

    # 初始化模型
    model = MineNet()

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 设置训练细节

    # 初始化学习率下降策略
    scheduler = StepLR(optimizer, step_size=5)

    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()

    # model.load_state_dict(torch.load('catdog_mineresnet_10.pth'))

    train_counter = []  # 训练集数量
    train_losses = []  # 训练集损失
    train_acces = []  # 训练集准确率
    val_counter = []  # 验证集数量
    val_losses = []  # 验证集损失
    val_acces = []  # 验证集准确率

    for epoch in range(1, EPOCHS + 1):
        # 刷新读取数据集
        train_loader, val_loader = refreshdataloader()
        # 开始训练并记录训练数据
        tr_loss, tr_acc = train(model, DEVICE, train_loader, optimizer, epoch)
        train_counter.append((epoch - 1) * len(train_loader.dataset))
        train_losses.append(tr_loss)
        train_acces.append(tr_acc)

        # 验证当前训练的预测效果
        vl, va = val(model, DEVICE, val_loader, epoch)
        val_counter.append((epoch - 1) * len(train_loader.dataset))
        val_losses.append(vl)
        val_acces.append(va)

        # 将当前批次模型保存下来
        filename_pth = f'./pth/catdog_mineresnet_{str(epoch)}.pth'
        torch.save(model.state_dict(), filename_pth)

    test(model, DEVICE, test_loader, 1)

    fig = plt.figure()  # 创建图像
    plt.plot(train_counter, train_losses, color='blue')  # 画出训练损失曲线
    plt.scatter(val_counter, val_losses, color='red')  # 画出测试损失散点图
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # 图例标识
    plt.xlabel('number of training examples seen')  # x轴标签
    plt.ylabel('negative log likelihood loss')  # y轴标签
    showDemo()
