from tasks.base_task import BaseTask
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SentimentDataset
from evaluation import mean_squared_error
import os
from shutil import copyfile
from transformers import get_linear_schedule_with_warmup
from utils.logging_utils import setup_logger
import numpy as np
import torch.nn.functional as f

logger = setup_logger('logs/SentimentModel')


class TrainingMultiRegressTask(BaseTask):
    def __init__(self, config, model):
        super().__init__(config, model)
        
        self.sentiment_loss = nn.MSELoss()
        
        self.load_datasets()
        self.create_dataloaders()

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=len(self.train_dataloader) * self.epoch
        )

        logger.info("%s start", config.TASK)
        logger.info("Learning Rate: %s", config.TRAINING.LEARNING_RATE)
        logger.info("Warm up: %s", self.warmup)
    
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
            best_val_score = np.inf
            patience = 0
        
        for it in range(self.epoch):
            logger.info('Epoch %s', self.running_epoch)
            self.train()
            self.evaluate_loss()
            
            # val scores
            scores = self.evaluate_metrics(self.dev_dataloader)
            logger.info('Scores: %s', scores)
            sentiment_val_score = scores[self.score]
            
            best = False
            if sentiment_val_score < best_val_score:
                best_val_score = sentiment_val_score
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

    def train(self):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with L1 Loss' % self.running_epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                outs = self.model(items['input_ids'],
                                  items['attention_mask'])
                
                # Apply sigmoid to the predictions
                outs = f.sigmoid(outs)
                
                self.optimizer.zero_grad()
                
                loss = self.sentiment_loss(outs,
                                           items['sentiment'])
                
                loss.backward()

                self.optimizer.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
        logger.info('Train Loss: %s', running_loss / len(self.train_dataloader))
        self.scheduler.step()

    def evaluate_loss(self):
        self.model.eval()
        running_loss = 0
        with tqdm(desc='Epoch %d - Validation' % self.running_epoch, unit='it', total=len(self.dev_dataloader)) as pbar:
            for it, items in enumerate(self.dev_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                    
                with torch.no_grad():
                    outs = self.model(items['input_ids'],
                                      items['attention_mask'])
                # Apply sigmoid to the predictions
                outs = f.sigmoid(outs)
                
                loss = self.sentiment_loss(outs,
                                           items['sentiment'])

                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
        logger.info('Dev Loss: %s', running_loss / len(self.dev_dataloader))

    def evaluate_metrics(self, dataloader):
        sentiment_gts, sentiment_gens = [], []
        
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.running_epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items['input_ids'],
                                      items['attention_mask'])
                
                outs = f.sigmoid(outs)

                sentiment_gens.append(outs)
                
                sentiment_gts.append(items['sentiment'])              
                
                pbar.update()
        sentiment_gts = torch.stack(sentiment_gts)
        sentiment_gens = torch.stack(sentiment_gens)

        sentiment_mse = mean_squared_error(sentiment_gens, sentiment_gts)

        scores = {
            'sentiment_mse': sentiment_mse,

        }

        print(scores)
        return scores
    
    def load_datasets(self):
        self.train_dataset = SentimentDataset(self.config.DATA,
                                              self.config.TRAINING.DATA_PATH.TRAIN)
        self.dev_dataset = SentimentDataset(self.config.DATA,
                                            self.config.TRAINING.DATA_PATH.DEV)
        self.test_dataset = SentimentDataset(self.config.DATA,
                                             self.config.TRAINING.DATA_PATH.TEST)
    
    def create_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.config.TRAINING.BATCH_SIZE,
                                           collate_fn=self.train_dataset.collate_fn)
        
        self.dev_dataloader = DataLoader(self.dev_dataset,
                                         batch_size=1,
                                         collate_fn=self.dev_dataset.collate_fn)
        
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=1,
                                          collate_fn=self.dev_dataset.collate_fn)