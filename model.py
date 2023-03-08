import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
import timm

# Language Encoder using phoBert model and get embedding of the question
class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super(LanguageEncoder, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])
        self.model = AutoModel.from_pretrained(config['pretrained_model'])

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output

# Image Encoder using Vision Transformer model and get embedding of the image
class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.config = config
        self.model = timm.create_model(config['pretrained_model'], pretrained=True)

    def forward(self, input_ids):
        output = self.model(input_ids)
        return output

# Coattention transformer using multimodal fusion modules: Coattention, Self-attention, and Cross-attention
class Coattention(nn.Module):
    def __init__(self, config):
        super(Coattention, self).__init__()
        self.config = config
        self.coattention = nn.Linear(config['coattention_dim'], config['coattention_dim'])
        self.self_attention = nn.Linear(config['coattention_dim'], config['coattention_dim'])
        self.cross_attention = nn.Linear(config['coattention_dim'], config['coattention_dim'])

    def forward(self, language_output, image_output):
        # Coattention
        language_output = language_output[0]
        image_output = image_output[0]
        language_output = self.coattention(language_output)
        image_output = self.coattention(image_output)
        language_output = language_output.transpose(1, 2)
        image_output = image_output.transpose(1, 2)
        coattention_output = torch.bmm(language_output, image_output)
        coattention_output = F.softmax(coattention_output, dim=2)
        coattention_output = torch.bmm(image_output, coattention_output)
        coattention_output = coattention_output.transpose(1, 2)
        coattention_output = self.coattention(coattention_output)
        coattention_output = coattention_output + language_output
        # Self-attention
        self_attention_output = self.self_attention(coattention_output)
        self_attention_output = self_attention_output.transpose(1, 2)
        self_attention_output = torch.bmm(self_attention_output, self_attention_output)
        self_attention_output = F.softmax(self_attention_output, dim=2)
        self_attention_output = torch.bmm(self_attention_output, coattention_output)
        self_attention_output = self_attention_output.transpose(1, 2)
        self_attention_output = self.self_attention(self_attention_output)
        self_attention_output = self_attention_output + coattention_output
        # Cross-attention
        cross_attention_output = self.cross_attention(self_attention_output)
        cross_attention_output = cross_attention_output.transpose(1, 2)
        cross_attention_output = torch.bmm(cross_attention_output, image_output)
        cross_attention_output = F.softmax(cross_attention_output, dim=2)
        cross_attention_output = torch.bmm(cross_attention_output, image_output)
        cross_attention_output = cross_attention_output.transpose(1, 2)
        cross_attention_output = self.cross_attention(cross_attention_output)
        cross_attention_output = cross_attention_output + self_attention_output
        return cross_attention_output

    

# VQA model combine Language Encoder and Image Encoder using Coattention transformer
class VQA(nn.Module):
    def __init__(self, config):
        super(VQA, self).__init__()
        self.config = config
        self.language_encoder = LanguageEncoder(config['language_encoder'])
        self.image_encoder = ImageEncoder(config['image_encoder'])
        self.coattention = Coattention(config['coattention'])

    def forward(self, input_ids, attention_mask, image):
        language_output = self.language_encoder(input_ids, attention_mask)
        image_output = self.image_encoder(image)
        output = self.coattention(language_output, image_output)
        return output



if __name__=="__main__":
    config = {
        'pretrained_model': 'vinai/phobert-base'
    }
    sentence = "Hôm nay trời đẹp quá"
    sentence = ViTokenizer.tokenize(sentence)
    model = LanguageEncoder(config)
    input_ids = torch.tensor([model.tokenizer.encode(sentence, add_special_tokens=True)])
    attention_mask = torch.tensor([[1]*len(input_ids[0])])
    output = model(input_ids, attention_mask)
    print(output)
    print(output[0].shape)
    print(output[1].shape)
    
