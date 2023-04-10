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


# Question dictionary:
question_dict = {
    "test_01":"màu của chiếc đĩa là gì",
    "test_02":"màu của xe buýt là gì",
    "test_03":"tủ lạnh bằng kim loại đặt ở đâu",
    "test_04":"người đàn ông đang ngồi ở đâu",
    "test_05":"có bao nhiêu người đàn ông cầm vợt",
}
# Image dictionary:
image_dict = {
    "test_01":"../viq_images/COCO_000000513241.jpg",
    "test_02":"../viq_images/COCO_000000140100.jpg",
    "test_03":"../viq_images/COCO_000000564912.jpg",
    "test_04":"../viq_images/COCO_000000544485.jpg",
    "test_05":"../viq_images/COCO_000000264884.jpg",
}

def callback(inference, test_select):
    # Display image
    st.write('You selected image:', image_dict[test_select])
    image = Image.open(image_dict[test_select])
    st.image(image, caption=image_dict[test_select], use_column_width=True)
    st.write('You selected question:', question_dict[test_select])

    question = question_dict[test_select]

    # Button predict:
    if st.button('Predict'):
        # Display answer:
        prediction_state = st.text('Model predicting...')
        st.write('Answer:', inference.predict(image, question))
        prediction_state = st.text('Model predicting...done')
@st.cache_resource
def load_model():
    config_path = "configs/base.yml"
    inference = Inference(config_path)
    return inference

if __name__=="__main__":
    st.title('ViVQA Demo') # Title of the app

    model_load_state = st.text('Loading config file and model...')
    config_path = "configs/base.yml"
    inference = load_model()
    model_load_state.text('Loading config file and model...done!')

    test_select = st.selectbox(
        'Select an test:',
        ('test_01', 'test_02', 'test_03', 'test_04', 'test_05'),
        key = 'test_select'
    )
    callback(inference, test_select)




