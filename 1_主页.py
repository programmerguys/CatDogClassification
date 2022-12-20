import streamlit as st
from streamlit_extras.app_logo import add_logo

# st.file_uploader('File uploader')

with open('readme.md', 'r') as f:
    st.markdown(f.read())

# Using "with" notation
with st.sidebar:
    st.markdown("# HaiPro 猫狗分类器")
    st.image("https://cdn.pixabay.com/photo/2017/10/06/12/30/dog-2822939_1280.jpg")
    st.text("这是一个猫狗分类器，可以识别猫和狗的图片。")
    st.text("作者：刘海")
