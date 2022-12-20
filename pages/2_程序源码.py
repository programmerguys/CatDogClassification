import streamlit as st
from streamlit_extras.buy_me_a_coffee import button
from streamlit_ace import st_ace


with st.sidebar:
    st.sidebar.header("嘿同学！在这里切换文件哦！")
    st.image("https://cdn.pixabay.com/photo/2017/10/06/12/30/dog-2822939_1280.jpg")
    st.text("这是一个猫狗分类器，可以识别猫和狗的图片。")
    st.text("作者：刘海")
    res = st.radio("请选择你要查看的源码文件：", ["训练模型源码", "猫狗游乐场源码"], index=0)
    button(username="lhcyqr", floating=False)



if res == "训练模型源码":
    st.title("猫狗分类器 - 模型训练源码展示")
    # Spawn a new Ace editor
    with open('train.py', 'r') as f:
        content = st_ace(
            value=f.read(),
            language='python',
            theme='ambiance',
            show_gutter=True,
            show_print_margin=True,
            wrap=True,
            auto_update=True,
        )
else:
    st.title("猫狗分类器 - 猫狗游乐场源码展示")
    with open('./pages/2_程序源码.py', 'r') as f:
        content = st_ace(
            value=f.read(),
            language='python',
            theme='ambiance',
            show_gutter=True,
            show_print_margin=True,
            wrap=True,
            auto_update=True,
        )