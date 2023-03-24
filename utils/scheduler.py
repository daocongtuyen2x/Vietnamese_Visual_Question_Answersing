from bisect import bisect_right
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau


class WarmupMultiStepLR(_LRScheduler):
    """
        Source: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py
    """
    def __init__(self, optimizer, milestones, iter_per_epoch, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of inreasing intergers. Got {}'.format(milestones))
        if warmup_method not in ('constant', 'linear'):
            raise ValueError("Only 'constant' or 'linaer' warmup_method accepted. Got {}".format(warmup_method))
        
        self.milestones = [m*iter_per_epoch for m in milestones]
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

        
    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch)/self.warmup_iters
                warmup_factor = self.warmup_factor * (1-alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs]
    
    def __call__(self, optimizer, i, epoch):
        self.step()
    
    
    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr


class WarmupCyclicalLR(object):
    """
        Cyclical learning rate scheduler with linear warm-up. E.g.:

        Step mode: ``lr = base_lr * 0.1 ^ {floor(epoch-1 / lr_step)}``.

        Cosine mode: ``lr = base_lr * 0.5 * (1 + cos(iter/maxiter))``.

        Poly mode: ``lr = base_lr * (1 - iter/maxiter) ^ 0.9``.
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=100, warmup_epochs=0):
        self.mode = mode 
        assert self.mode in ('cos', 'poly', 'step'),  'Unsupported learning rate scheduler'
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = int(warmup_epochs * iters_per_epoch)
    
    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        
        # warm-up lr scheduler
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
            
        assert lr > 0
        self._adjust_learning_rate(optimizer, lr)
    
    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
            
class _MultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones=[15, 30, 45], gamma=0.5, start_epoch=0):
        self.epoch = start_epoch
        super(_MultiStepLR, self).__init__(optimizer, milestones=[15, 30, 45], gamma=0.5)
        
    def __call__(self, optimizer, i, epoch):
        if epoch == self.epoch + 1:
            self.step()
            self.epoch += 1
        
            
def build_scheduler(cfg, optimizer, iter_per_epoch=1500, start_epoch=0):
    if cfg['scheduler']['type'] == 'cyclical':
        scheduler = WarmupCyclicalLR(
            mode=cfg['scheduler']['args']['mode'], 
            base_lr=cfg['optimizer']['args']['lr'], 
            num_epochs=cfg['train_params']['n_epochs'], 
            iters_per_epoch=iter_per_epoch, 
            warmup_epochs=cfg['scheduler']['args']['warmup_epochs'])
        
    elif cfg['scheduler']['type'] == 'step':
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer, 
            milestones=cfg['scheduler']['args']['milestones'], 
            iter_per_epoch=iter_per_epoch,
            warmup_factor=cfg['optimizer']['args']['lr']/(cfg['scheduler']['args']['warmup_epochs'] * iter_per_epoch), 
            warmup_iters=cfg['scheduler']['args']['warmup_length'] * iter_per_epoch)
    elif cfg['scheduler']['type'] == 'multistep':
        scheduler = _MultiStepLR(
            optimizer=optimizer, 
            milestones=cfg['scheduler']['args']['milestones'],
            start_epoch=start_epoch
        )
    elif cfg['scheduler']['type'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    else:
        scheduler = None
#         scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False)

    return scheduler