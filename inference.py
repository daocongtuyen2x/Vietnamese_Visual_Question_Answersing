import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import *
from dataset import TextProcessing
from model import ViVQANet
import yaml
from transformers import AutoTokenizer
from PIL import Image
import json

class Inference():
    def __init__(self, cfg_path, device='cpu'):
        super(Inference, self).__init__()
        self.cfg_path = cfg_path
        with open(self.cfg_path) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.device = device
        self.model = ViVQANet(self.cfg)
        if self.cfg['inference'] is not None:
            self.model.load_state_dict(torch.load(
                self.cfg['inference']['weight_path'], map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        print('Model loaded!')

        self.image_transform = clip_transform(size=224)
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.question_transform = TextProcessing(
            self.tokenizer, max_seq_len=40)
        with open(self.cfg['data_params']['label_dict_path'], 'r') as fp:
            self.label_dict = json.load(fp)

        self.idx2label = {v: k for k, v in self.label_dict.items()}

    def preprocess(self, image, question):
        image = self.image_transform(image)
        input_ids, attention_mask, _, _ = self.question_transform.process_data(
            question)
        image = image.unsqueeze(0)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        batch = {
            'image_tensor': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return batch

    def predict(self, image, question):
        batch = self.preprocess(image, question)
        with torch.no_grad():
            output = self.model(batch)
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            output = self.idx2label[output.item()]
        return output


if __name__ == "__main__":
    config_path = "configs/base.yml"
    image_path = "../viq_images/COCO_000000022440.jpg"
    question = "màu của xe buýt là gì"
    image = Image.open(image_path)
    inference = Inference(config_path)
    output = inference.predict(image, question)
    print('image_path:', image_path)
    print('question: ', question)
    print('answer: ', output)
    
