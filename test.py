from utils.transforms import *
import yaml
from PIL import Image
import json

from inference import Inference
from metric import wup_measure
import pandas as pd
import os
from tqdm import tqdm


def test():
    config_path = "configs/base.yml"
    inference = Inference(config_path)
    df = pd.read_csv('data/test.csv')[1417:]
    print('len df:', len(df))
    wups_00 = []
    wups_09 = []
    true_count = 0
    print('Load pretrained model successfully!')
    for i, row in tqdm(df.iterrows()):
        answer = row['answer']
        question = row['question']
        image_id = row['img_id']
        image_path = os.path.join('../viq_images', "COCO_"+"0"*(12-len(str(image_id)))+str(image_id)+".jpg")
        image = Image.open(image_path)
        pred_answer = inference.predict(image, question)
        wups_00.append(wup_measure(answer, pred_answer, 0.0))
        wups_09.append(wup_measure(answer, pred_answer, 0.9))
        if answer == pred_answer:
            true_count += 1
    print('wup_00: ', sum(wups_00)/len(wups_00))
    print('wup_09: ', sum(wups_09)/len(wups_09))
    print('accuracy: ', true_count/len(df))

if __name__ == "__main__":
    test()




