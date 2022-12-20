import os
import random

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models  # PyTorch数据转换、预训练模型
from train import MineNet, labelList
import torch
import torchvision
from PIL import Image
from io import BytesIO
import glob

st.title("猫狗分类器 - 游乐场")
st.subheader("一、你可以通过选择以下随机六张中的一张图片来测试")
cPath = os.getcwd()  # 获取当前路径
train_dir = f'{cPath}/data/train'
test_dir = f'{cPath}/data/test'
train_files = os.listdir(train_dir)  # 训练集文件名下的所有文件
test_files = os.listdir(test_dir)  # 测试集文件名下的所有文件
# 把猫的图片和狗的图片分开
cat_files = [tf for tf in train_files if 'cat' in tf]
dog_files = [tf for tf in train_files if 'dog' in tf]

className = {
    'cat': '「小猫🐱」',
    'dog': '「小狗🐶」',
}


def predict(model, image_bytes):
    # 将字节集图片转换为可读取的图片对象
    image = Image.open(BytesIO(image_bytes))

    # 对图片进行预处理
    transform = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小
        transforms.CenterCrop(224),  # 中心裁剪图像
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])
    image = transform(image).unsqueeze(0)  # 扩展维度并转换为张量

    # 将图片输入到模型中进行预测
    output = model(image)

    # 处理预测结果
    _, pred = torch.max(output, 1)  # 获取预测类别的索引
    return labelList[pred]  # 获取类别的字符串名称


# 对文件进行降序排序
def extract_number(filename):
    # 从文件名中提取数字
    return int(filename.split('_')[-1].split('.')[0])


with st.sidebar:
    st.sidebar.header("嘿同学！在这里切换模型哦！")
    # 读取所有pth文件
    files = glob.glob('./pth/catdog_mineresnet_*.pth')

    # 对文件进行降序排序
    files = sorted(files, key=extract_number, reverse=True)

    # 打印结果
    print(files)

    res_radio = st.radio("请选择你要使用的模型：", files, index=0)

    models_path = files[0] if res_radio is None else res_radio
    # 加载模型
    model = MineNet()
    model.load_state_dict(torch.load(models_path))
    model.eval()  # 将模型设置为评估模式

    st.text("模型加载完成！")
    st.text("你可以通过上传图片或者选择图片来测试模型！")


def show_result(file_data, choose_type="上传"):
    st.sidebar.header(f"嘿同学！这里显示{choose_type}的结果哦！")
    # 将字节集数据转换为可读取的文件对象
    image_file = BytesIO(file_data)
    image = Image.open(image_file)
    pred_class = predict(model, file_data)
    print(pred_class)  # 输出预测的类别
    st.sidebar.image(image, caption=f"这是你{choose_type}的图片，我认为它是{className[pred_class]}")


def demo_on_click(filepath):
    print(f"demo_on_click: {filepath}")
    with open(filepath, "rb") as f:
        show_result(f.read(), "选择")


def render_item(item_col, _filename):
    with item_col:
        _typename = '猫 🐱' if 'cat' in _filename else '狗 🐶'
        _typename += _filename
        st.image(os.path.join(train_dir, _filename), caption=_typename, width=200)
        st.button(f"就选它了！{'🐱 喵呜！' if 'cat' in _filename else '🐶 汪汪！'}", on_click=demo_on_click,
                  args=(os.path.join(train_dir, _filename),),
                  key=_filename)


def render_list(_type: str = "all"):
    if _type == "cat":
        _filenameList = random.choices(cat_files, k=6)
    elif _type == "dog":
        _filenameList = random.choices(dog_files, k=6)
    else:
        _filenameList = random.choices(train_files, k=6)

    index = 0
    for _ in range(2):
        col1, col2, col3 = st.columns(3)
        render_item(col1, _filenameList[index])
        render_item(col2, _filenameList[index + 1])
        render_item(col3, _filenameList[index + 2])
        index += 3


tab_all, tab_cat, tab_dog = st.tabs(["所有动物", "🐱只要小猫", "🐶只要小狗"])

with tab_all:
    render_list("all")

with tab_cat:
    render_list("cat")

with tab_dog:
    render_list("dog")

st.subheader("二、你也可以上传一张小猫🐱或者小狗🐶的图片")
uploaded_file = st.file_uploader("请选择一张阿猫阿狗的照片~", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    print(uploaded_file)
    show_result(uploaded_file.read(), "上传")
