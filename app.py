import streamlit as st 
import os
from WaterNet.WaterNet_test import WaterNet_test
from UWCNN.UWCNN_test import UWCNN_test
import time

# streamlit run app.py

# 设置缓存文件夹
cache_dir = 'cache'
if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

# 设置全局属性
st.set_page_config(
    page_title='水下图像增强系统',
    page_icon=' ',
    layout='wide'
)

# 设置侧边栏
with st.sidebar:
    st.title('欢迎来到水下图像增强系统')
    file_uploader = st.file_uploader(label='上传所需处理的图片', type=['.png','.bmp','jpg'])
    st.markdown('---')
    mode = st.radio(label='模型', options=['UWCNN', 'WaterNet'])
    detect = st.button(label='开始处理')

# 设置文件加载
if file_uploader:
    file_name = file_uploader.name
    input_cache_path = os.path.join(cache_dir, file_name)
    open(input_cache_path, 'wb').write(file_uploader.read())

# 设置展示框
p1, p2 = st.columns(spec=2)
p1.title('Input-picture')
p2.title('Output-picture')

# 显示加载的图片
if file_uploader and os.path.exists(input_cache_path):
    p1.image(image=input_cache_path, caption='input image', use_column_width='always')
else:
    p1.info('等待图片加载...')

# 开始检测
if detect and os.path.exists(input_cache_path):
    #调用test程序
    start_time = time.time()
    if mode == 'UWCNN':
        name = UWCNN_test(file_name)
        p2.image(image=f'UWCNN\\output\\{name}', caption='output image', use_column_width='always')
    elif mode == 'WaterNet':
        name = WaterNet_test(file_name)
        p2.image(image=f'WaterNet\\output\\{name}', caption='output image', use_column_width='always')
    end_time = time.time()
    last_time = '%.2f'%(end_time-start_time)
    st.info(f'处理耗时:{last_time}s')
else:
    p2.info('等待处理...')

