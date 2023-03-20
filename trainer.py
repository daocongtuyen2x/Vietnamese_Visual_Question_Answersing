import os
import numpy as np
import torch
from tqdm import tqdm
import time
from collections import OrderedDict
from fastprogress import progress_bar
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from utils.optimizer import build_optim
from utils.scheduler import build_scheduler
import gc
from model import ViVQANet

class Trainer:

    def __init__(self, cfg, device='cpu'):
        model = ViVQANet(cfg)
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        # self.logger = logger
        # self.writer = writer
        self.optimizer = build_optim(cfg['optimizer'], self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.criterion = torch.nn.CrossEntropyLoss()

        
    def train(self, epoch, dataloader):
        start_time = time.time()
        self.model.train()
        mean_loss = []
        mean_acc = []
        for step, batch in tqdm(enumerate(dataloader)):
            # img = batch['image_tensor'].to(self.device)
            # input_ids = batch['input_ids'].to(self.device)
            # attention_mask = batch['attention_mask'].to(self.device)
            label = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(batch)
                loss = self.criterion(logits, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            acc = (logits.argmax(1) == label).float().mean()
            mean_loss.append(loss.item())
            mean_acc.append(acc.item())
            # tbar.set_description(f'Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}')
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info(f'Epoch {epoch} | Loss: {np.mean(mean_loss):.4f} | Acc: {np.mean(mean_acc):.4f} | Time: {time.time() - start_time:.2f}s')
        # self.writer.add_scalar('train/loss', np.mean(mean_loss), epoch)
        del loss, logits, img, input_ids, attention_mask, label
        return np.mean(mean_loss), np.mean(mean_acc)
    @torch.no_grad()
    def test(self, epoch, dataloader):
        start_time = time.time()
        self.model.eval()
        mean_loss = []
        mean_acc = []
        # tbar = progress_bar(dataloader)
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader)):
                # img = batch['image_tensor'].to(self.device)
                # input_ids = batch['input_ids'].to(self.device)
                # attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, label)
                acc = (logits.argmax(1) == label).float().mean()
                mean_loss.append(loss.item())
                mean_acc.append(acc.item())
                # tbar.set_description(f'Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}')
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info(f'Epoch {epoch} | Loss: {np.mean(mean_loss):.4f} | Acc: {np.mean(mean_acc):.4f} | Time: {time.time() - start_time:.2f}s')
        # self.writer.add_scalar('test/loss', np.mean(mean_loss), epoch)
        del loss, logits, img, input_ids, attention_mask, label
        return np.mean(mean_loss), np.mean(mean_acc)

    @staticmethod        
    def save_checkpoint(state, root, filename):
        save_dir = os.path.join(root, filename)
        torch.save(state, save_dir)

    @staticmethod
    def load_checkpoint(path, model):
        start_epoch = 0
        best_metric = 0.0
        if os.path.exists(path):
            ckpt = torch.load(path, 'cpu')
            model.load_state_dict(ckpt['state_dict'])
            start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
        
        print(f"{'='*5} Load checkpoint at epoch {start_epoch} with Dice global {best_metric} {'='*5}")
        return model, start_epoch, best_metric