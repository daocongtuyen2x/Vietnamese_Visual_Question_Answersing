import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
from pyvi import ViTokenizer
from torchvision import transforms
import json
# from .utils.utils import get_transforms

class TextProcessing(object):
    def __init__(self, tokenizer, max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.word_segmenter = ViTokenizer

    def process_data(self, text):
        # Setting based on the current model type
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        unk_token = self.tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id

        cls_token_segment_id=0
        pad_token_segment_id=0
        sequence_a_segment_id=0
        mask_padding_with_zero=True
        
        
        # Tokenize word by word
        tokens = []
        for word in text.split():
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens.extend(word_tokens)
        
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > self.max_seq_len - special_tokens_count:
            tokens = tokens[: (self.max_seq_len - special_tokens_count)]
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([pad_token_segment_id] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == self.max_seq_len, "Error with input length {} vs {}".format(len(input_ids), self.args.max_seq_len)
        assert len(attention_mask) == self.max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), self.max_seq_len
        )
        assert len(token_type_ids) == self.max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), self.max_seq_len
        )
        
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, tokens

class ViVQA_Dataset(Dataset):
    def __init__(self, df_path, label_dict_path, tokenizer, max_len=256, image_dir=None, transform=None):
        self.df = pd.read_csv(df_path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.processor = TextProcessing(self.tokenizer, self.max_len)
        self.image_dir = image_dir
        self.transform = transform

        with open(label_dict_path, 'r') as fp:
            self.label_dict = json.load(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        question = self.df.iloc[idx]['question']
        image_id = self.df.iloc[idx]['img_id']
        answer = self.df.iloc[idx]['answer']
        label = self.label_dict[answer]

        # process text data
        input_ids, attention_mask, _, _ = self.processor.process_data(question)

        # process image data
        image_path = os.path.join(self.image_dir, "COCO_"+"0"*(12-len(str(image_id)))+str(image_id)+".jpg")
        image_tensor = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image_tensor)
        else:
            image_tensor = torch.tensor(image_tensor).type(torch.long)
        label = torch.tensor(label).type(torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_tensor": image_tensor,
            "label": label
        }

def get_dataloader(df_path, label_dict_path, tokenizer, max_len=256, image_dir=None, transform=None, batch_size=8, shuffle=True):
    dataset = ViVQA_Dataset(df_path, label_dict_path, tokenizer, max_len, image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__=="__main__":
    img_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    test_dataset = ViVQA_Dataset(
        df_path = 'data/test.csv', 
        label_dict_path = 'data/label_dict.json',
        tokenizer = tokenizer,
        max_len=256,
        image_dir="../viq_images",
        transform=img_transforms
    )
    dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['image_tensor'].shape)
        print(batch['label'].shape)
        break







        



        

