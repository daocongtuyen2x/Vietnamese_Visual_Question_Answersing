from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn as nn
import logging
import os


def get_transforms():
    return Compose([
        Resize(224),
        # Resize(256),
        # CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def init_logger(log_file='logs/log.txt'):
    logger = logging.getLogger()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)

    if not os.path.exists(log_file):
        open(log_file, 'a')
    
    fhandler = logging.FileHandler(filename=log_file, mode='a')

    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.INFO)
    logger.addHandler(fhandler)
    logger.info(f"{'='*20} ViVQA {'='*20}")
    return logger