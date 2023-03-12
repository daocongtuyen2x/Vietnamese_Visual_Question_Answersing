import torch
from adan_pytorch import Adan

def build_optim(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg['args']['lr']
        weight_decay = cfg['args']['weight_decay']
        if 'bias' in key:
            weight_decay = cfg['args']['weight_decay_bias']
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
    
    if cfg['type'] == 'adam':
        optimizer = torch.optim.AdamW(params, lr, eps=1e-6)
    elif cfg['type'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)
    elif cfg['type'] == 'adan':
        optimizer = Adan(params, lr=lr, betas = (0.02, 0.08, 0.01), weight_decay = 0.003)
    return optimizer