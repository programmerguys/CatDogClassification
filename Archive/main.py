# 导入库
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

# 设置超参数
# 每批次的大小数量
BATCH_SIZE = 20
# 迭代次数
EPOCHS = 10
# 采用cpu还是gpu进行计算，如果gpu能用就用gpu，否则用cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose(
    [transforms.Resize(100),
     transforms.RandomVerticalFlip(),
     transforms.RandomCrop(50),
     transforms.RandomResizedCrop(150),
     transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ])
# 读取数据
dataset_train = datasets.ImageFolder('./dc_2000/train', transform)
dataset_test = datasets.ImageFolder('./dc_2000/test', transform)

print(dataset_train.imgs)

# 对应文件夹的label
print(dataset_train.class_to_idx)

# 对应文件夹的label
print(dataset_test.class_to_idx)

# 导入数据，从10000张图片中随机选取20张图片作为一个批次，乱序随机抽取。
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


# 定义网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)  # ReLU函数被定义为该元素与0的最大值
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


modellr = 1e-4  # 学习率 1的-4次方 0.0001

# 实例化模型并且移动到GPU
model = ConvNet().to(DEVICE)

# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=modellr)


# 调整学习率
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("lr:", modellrnew)  # 打印学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        output = model(data)

        # print(output)

        loss = F.binary_cross_entropy(output, target)

        loss.backward()

        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


# 定义测试过程
def val(model, device, test_loader):
    model.eval()  # 把模型转为test模式
    test_loss = 0  # 定义测试损失
    correct = 0  # 定义正确的个数

    with torch.no_grad():  # 不计算梯度，节省内存
        for data, target in test_loader:  # 循环测试集
            data, target = data.to(device), target.to(device).float().unsqueeze(1)  # 把数据和标签移动到GPU
            output = model(data)  # 把数据输入网络得到输出
            # print(output)
            # 二元交叉熵损失函数
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()  # mean()取平均值
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)  # 把输出转换为0或者1 即猫或狗
            correct += pred.eq(target.long()).sum().item()  # 计算正确的个数

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,  # 平均损失
            correct,  # 正确的个数
            len(test_loader.dataset),  # 测试集的总数
            100. * correct / len(test_loader.dataset)  # 正确率
        ))


# 训练
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)  # 调整学习率
    train(model, DEVICE, train_loader, optimizer, epoch)  # 训练
    val(model, DEVICE, test_loader)  # 测试

# 保存模型
torch.save(model, '../model.pth')

# 模型存储路径
model_save_path = '../model.pth'

# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
# 数据预处理
transform_test = transforms.Compose([
    transforms.Resize(100),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(50),
    transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class_names = ['cat', 'dog']  # 这个顺序很重要，要和训练时候的类名顺序一致

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------ 载入模型并且训练 --------------------------- #
model = torch.load(model_save_path)
model.eval()
# print(model)

image_PIL = Image.open('../dc_2000/test/cat/cat.1499.jpg')
image_tensor = transform_test(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 没有这句话会报错
image_tensor = image_tensor.to(device)

out = model(image_tensor)
pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(device)
print(class_names[pred])
