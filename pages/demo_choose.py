import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import os
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from inference import Inference

st.set_page_config(page_title="Demo in Test set", page_icon="📈")


# Question dictionary:
question_dict = {
    "test_01":"màu của xe buýt là gì",
    "test_02":"màu của con ngựa là gì",
    "test_03":"màu của xe máy là gì",
    "test_04":"màu của vòi nước là gì",
    "test_05":"màu của gấu bông là gì",
    "test_06":"người đang nằm ở đâu",
    "test_07":"con mèo đen trắng nằm ở đâu",
    "test_08":"con mèo đang ngủ trưa ở đâu",
    "test_09":"người phụ nữ đặt chảo ở đâu",
    "test_10":"người phụ nữ đang chụp ảnh mình ở đâu và con chó của mình ở đâu",
    "test_11":"có gì bay cao trên bầu trời",
    "test_12":"con vật ba người đứng trên là gì",
    "test_13":"những gì giữ một cái đèn trên đỉnh của đồng hồ",
    "test_14":"những gì được hiển thị với thức ăn bên trong",
    "test_15":"những gì đang ngồi trên đôi giày ngoài trời",
    "test_16":"có bao nhiêu người bạn đang nằm trên giường trong phòng khách sạn",
    "test_17":"có bao nhiêu điện thoại nằm trên một tấm chăn màu cam",
    "test_18":"có bao nhiêu con ngựa vằn đứng gần một vùng nước",
    "test_19":"có bao nhiêu cây gậy bóng chày gỗ xếp trên bàn",
    "test_20":"có bao nhiêu con vịt cùng nhau bơi qua vùng nước",

}
image_dict = {
    "test_01":"../viq_images/COCO_000000140100.jpg",
    "test_02":"../viq_images/COCO_000000083151.jpg",
    "test_03":"../viq_images/COCO_000000219254.jpg",
    "test_04":"../viq_images/COCO_000000155376.jpg",
    "test_05":"../viq_images/COCO_000000406030.jpg",
    "test_06":"../viq_images/COCO_000000570801.jpg",
    "test_07":"../viq_images/COCO_000000481465.jpg",
    "test_08":"../viq_images/COCO_000000084609.jpg",
    "test_09":"../viq_images/COCO_000000419980.jpg",
    "test_10":"../viq_images/COCO_000000024247.jpg",
    "test_11":"../viq_images/COCO_000000384476.jpg",
    "test_12":"../viq_images/COCO_000000569379.jpg",
    "test_13":"../viq_images/COCO_000000545007.jpg",
    "test_14":"../viq_images/COCO_000000455974.jpg",
    "test_15":"../viq_images/COCO_000000555705.jpg",
    "test_16":"../viq_images/COCO_000000100579.jpg",
    "test_17":"../viq_images/COCO_000000445906.jpg",
    "test_18":"../viq_images/COCO_000000506640.jpg",
    "test_19":"../viq_images/COCO_000000528432.jpg",
    "test_20":"../viq_images/COCO_000000442069.jpg",
}

def callback(inference, test_select):
    # Display image
    st.write('You selected image:', image_dict[test_select])
    image = Image.open(image_dict[test_select])
    
    st.image(image, caption=image_dict[test_select], use_column_width=True)
    st.write('You selected question:', question_dict[test_select] + '?')

    question = question_dict[test_select]

    # Button predict:
    if st.button('Predict'):
        # Display answer:
        with st.spinner('Loading model...'):
            st.write('Answer:', inference.predict(image, question))
        st.success('Done!')
@st.cache_resource
def load_model():
    config_path = "configs/base.yml"
    inference = Inference(config_path)
    return inference

if __name__=="__main__":
    st.title('ViVQA Demo') # Title of the app

    config_path = "configs/base.yml"
    inference = load_model()
    st.success('Model loaded!')

    test_select = st.selectbox(
        'Select a test case:',
        ('test_01', 'test_02', 'test_03', 'test_04', 'test_05', 'test_06', 'test_07', 'test_08', 'test_09', 'test_10', 'test_11', 'test_12', 'test_13', 'test_14', 'test_15', 'test_16', 'test_17', 'test_18', 'test_19', 'test_20'),
        key = 'test_select'
    )
    callback(inference, test_select)




