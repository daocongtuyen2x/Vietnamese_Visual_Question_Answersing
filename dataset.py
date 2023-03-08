import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd

class ProcessData(object):
    def __init__(self, tokenizer, max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

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
        
        
        # Tokenize word by word (for NER)
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


class EncodeLabel(object):
    def __init__(self, question_id):
        self.question_id = question_id
    
    def encode(self, label):
        return torch.tensor([1])

class VQA_Dataset(Dataset):
    def __init__(self, df, tokenizer, question_type, max_len=256, image_dir=None, transform=None):
        self.question_type = question_type
        self.df = df[df['type']==self.question_id]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processor = ProcessData(self.tokenizer, self.max_len)
        self.label_encoder = EncodeLabel()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # process text data
        question = self.data[idx]['question']
        image_id = self.data[idx]['img_id']
        answer = self.data[idx]['answer']
        input_ids, attention_mask, _, _ = self.processor.process_data(question)
        encode_label = self.label_encoder.encode(answer)

        # process image data
        image_path = os.path.join(self.image_dir, "COCO_"+"0"*(12-len(str(image_id)))+str(image_id)+".jpg")
        image_tensor = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image_tensor)
        else:
            image_tensor = torch.tensor(image_tensor).type(torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encode_label": encode_label,
            "image_tensor": image_tensor,
        }







        



        

