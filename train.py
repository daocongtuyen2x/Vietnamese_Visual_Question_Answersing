import os
import time
import yaml
import shutil
import numpy as np
import argparse
import torch
from torchvision import transforms
from dataset import get_dataloader

from trainer import Trainer
from transformers import AutoModel, AutoTokenizer

if __name__=="__main__":
    args = argparse.ArgumentParser(description='Main file')
    args.add_argument('--config', default='./configs/base.yml', type=str,
                      help='config file path (default: None)')
    # 1. Read experiment configurations
    cmd_args = args.parse_args()
    nn_config_path = cmd_args.config
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    # 2. Create experiment folder
    save_dir = f'experiments/{cfg["name"]}-{cfg["model_params"]["network"]}-{cfg["train_params"]["n_epochs"]}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # 3. Save config file
    shutil.copy(nn_config_path, save_dir)
    # 4. Load dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_transforms = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    train_loader = get_dataloader(
        cfg['data_params']['train_csv_path'], 
        cfg['data_params']['label_dict_path'], 
        tokenizer, 
        max_len=40, 
        image_dir=cfg['data_params']['image_dir'], 
        transform=img_transforms, 
        batch_size=cfg['train_params']['train_batch_size'], 
        shuffle=True
    )
    test_loader = get_dataloader(
        cfg['data_params']['test_csv_path'], 
        cfg['data_params']['label_dict_path'], 
        tokenizer, 
        max_len=40, 
        image_dir=cfg['data_params']['image_dir'], 
        transform=img_transforms, 
        batch_size=cfg['train_params']['train_batch_size'], 
        shuffle=True
    )
    # 4. Create trainer
    trainer = Trainer(cfg, device)
    # 5. Train
    start_epoch = 1
    n_epochs = cfg['train_params']['n_epochs']
    for epoch in range(start_epoch, n_epochs+1):
        print(f"{'='*10} Epoch: {epoch}/{n_epochs} {'='*10}")
        train_loss, train_acc = trainer.train(epoch, train_loader)
        print('Train | Loss: {:.4f} | Acc: {:.4f}'.format(train_loss, train_acc))