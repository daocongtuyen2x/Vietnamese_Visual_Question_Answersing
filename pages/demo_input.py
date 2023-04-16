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

st.set_page_config(page_title="Demo input question", page_icon="📈")

image_dict = {
    "test_01":"../viq_images/COCO_000000140100.jpg", # xe buýt
    "test_02":"../viq_images/COCO_000000262201.jpg", # con chim
    "test_03":"../viq_images/COCO_000000132646.jpg", # chai nhựa
    "test_04":"../viq_images/COCO_000000262012.jpg", # chiếc ô
    "test_05":"../viq_images/COCO_000000079971.jpg", # chiếc lá + hoa, màu xe máy: ../viq_images/COCO_000000476791.jpg
    "test_06":"../viq_images/COCO_000000558555.jpg", # hộp
    "test_07":"../viq_images/COCO_000000573759.jpg", # người đàn ông và 3 con chó đang trên chiếc thuyền
    "test_08":"../viq_images/COCO_000000106315.jpg", # bát
    "test_09":"../viq_images/COCO_000000098298.jpg", # nhà vệ sinh, phòng tắm
    "test_10":"../viq_images/COCO_000000069346.jpg", # bánh pizza ở đâu
    "test_11":"../viq_images/COCO_000000342682.jpg", # ngựa vằn
    "test_12":"../viq_images/COCO_000000466456.jpg", # cô dâu và chú rể cắt bánh
    "test_13":"../viq_images/COCO_000000341048.jpg", # gấu trúc
    "test_14":"../viq_images/COCO_000000425475.jpg", # ly
    "test_15":"../viq_images/COCO_000000555705.jpg",
    "test_16":"../viq_images/COCO_000000219723.jpg", # có bao nhiêu hộp chứa gà con 
    "test_17":"../viq_images/COCO_000000238843.jpg", # có bao nhiêu con hươu cao cổ
    "test_18":"../viq_images/COCO_000000271401.jpg", # có bao nhiêu chiếc máy bay
    "test_19":"../viq_images/COCO_000000092729.jpg", # có bao nhiêu con mèo
    "test_20":"../viq_images/COCO_000000571503.jpg", # có bao nhiêu máy bay
}

def callback(inference, test_select):
    # Display image
    st.write('You selected image:', image_dict[test_select])
    image = Image.open(image_dict[test_select])
    
    st.image(image, caption=image_dict[test_select], use_column_width=True)
    
    question = st.text_input('Enter question (Vietnamese): ', 'Màu của con vật là gì ?')
    if "?" in question:
        question = question.replace("?", "")

    # Button predict:
    if st.button('Predict'):
        # Display answer:
        with st.spinner('Wait for it...'):
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
        'Select an test:',
        ('test_01', 'test_02', 'test_03', 'test_04', 'test_05', 'test_06', 'test_07', 'test_08', 'test_09', 'test_10', 'test_11', 'test_12', 'test_13', 'test_14', 'test_15', 'test_16', 'test_17', 'test_18', 'test_19', 'test_20'),
        key = 'test_select'
    )

    callback(inference, test_select)
