import torch
from ViVQA.modules.dataset import VQA_Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class Config:
    def __init__(self):
        self.language_encoder = {
            'pretrained_model': 'vinai/phobert-base',
            'hidden_size': 768,
        }
        self.image_encoder = {
            'pretrained_model': 'vit_base_patch16_384',
            'hidden_size': 768,
        }
        self.hidden_size = 768
        self.dropout = 0.1
        self.num_classes = 1
        

if __name__=="__main__":
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.language_encoder['pretrained_model'])
    dataset = VQA_Dataset('data', tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['image'].shape)
        break