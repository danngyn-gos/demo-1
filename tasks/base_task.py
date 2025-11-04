import torch
from torch import nn, optim
from shutil import copyfile
from torch.optim.lr_scheduler import LambdaLR
import os
from abc import ABC, abstractmethod
import numpy as np


class BaseTask(ABC):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.patience = config.TRAINING.PATIENCE
        self.device = config.TRAINING.DEVICE
        self.score = config.TRAINING.SCORE
        self.warmup = config.TRAINING.WARMUP
        self.checkpoint_path = config.TRAINING.CHECKPOINT_PATH
        
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=float(config.TRAINING.LEARNING_RATE),
                                    betas=(0.9, 0.98))

        self.epoch = config.TRAINING['EPOCH']
        self.running_epoch = 0    

    @abstractmethod
    def train(self):
        raise NotImplementedError 

    @abstractmethod
    def evaluate_loss(self):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_metrics(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_datasets(self):
        raise NotImplementedError
    
    @abstractmethod
    def create_dataloaders(self):
        raise NotImplementedError
    
    def save_checkpoint(self, dict_for_updating):
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        dict_for_saving = {
            'epoch': self.running_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value
        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None
        checkpoint = torch.load(fname)

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        return checkpoint

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(
                os.path.join(
                    self.checkpoint_path,
                    "last_model.pth"
                    )
                )
            # use_rl = checkpoint["use_rl"]
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.running_epoch = checkpoint["epoch"] + 1
            self.epoch = self.epoch - self.running_epoch
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        else:
            best_val_score = .0
            patience = 0
        
        for it in range(self.epoch):
            self.train()
            self.evaluate_loss()
            
            # val scores
            scores = self.evaluate_metrics(self.dev_dataloader)
            val_score = scores[self.score]

            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                exit_train = True

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                         os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.running_epoch += 1
        test_scores = self.evaluate_metrics(self.test_dataloader)
        print(f"Evaluation on test set: {test_scores}")
