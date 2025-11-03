from base_task import BaseTask
import torch
from torch import nn
from torch.utils.data import DataLoader
from training_task.base_task import BaseTask
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from dataset.anger_toxic_dataset import AngerToxicDataset
from evaluation import accuracy


class TrainingMultiTask(BaseTask):
    def __init__(self, config, model):
        super().__init__(config, model)

        self.scheduler = None
        self.anger_loss_fn = nn.CrossEntropyLoss()
        self.toxic_loss_fn = nn.CrossEntropyLoss(weight=config.TRAINING.TOXIC_WEIGHTED_LOSS)
        self.scheduler = self.LambdaLR(self.optimizer, self.lambda_lr)
        
    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.latent_dim_mlp ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        running_loss = 0
        with tqdm(desc='Epoch %d - Training with Cross Entropy Loss' % self.running_epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                out = self.model(items['input_ids'], items['attention_mask'])
                
                self.optimizer.zero_grad()
                
                anger_loss = self.anger_loss_fn(out['anger_output'], items['anger'])
                toxic_loss = self.toxic_loss_fn(out['toxic_output'], items['toxic'])
                
                loss = anger_loss + toxic_loss
                
                loss.backward()

                self.optimizer.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
        self.scheduler.step()


    def evaluate_loss(self):
        self.model.eval()
        running_loss = 0
        with tqdm(desc='ValidationEpoch %d - Validation' % self.running_epoch, unit='it', total=len(self.dev_dataloader)) as pbar:
            for it, items in enumerate(self.dev_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                    
                    with torch.no_grad():
                        out = self.model(items['input_ids'],
                                         items['attention_mask'])
                
                self.optimizer.zero_grad()
                
                anger_loss = self.anger_loss_fn(out['anger_output'], items['anger'])
                toxic_loss = self.toxic_loss_fn(out['toxic_output'], items['toxic'])
                
                loss = anger_loss + toxic_loss

                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                
    def evaluation(self):
        anger_gts, toxic_gts = [], []
        anger_gens, toxic_gens = [], []
        
        self.model.eval()
        with tqdm(desc='Epoch %d - Evaluation' % self.running_epoch, unit='it', total=len(self.test_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                for key, value in items.items():
                    if isinstance(value, torch.Tensor):
                        items[key] = value.to(self.device)
                with torch.inference_mode():
                    outs = self.model(items['input_ids'],
                                      items['attention_mask'])
                
                anger_gts.append(items['anger'])
                anger_gens.append(outs['angry_output'])

                toxic_gts.append(items['toxic'])
                toxic_gens.append(outs['toxic_output'])
                
                pbar.update()
        anger_gts = torch.stack(anger_gts)
        anger_gens = torch.stack(anger_gens)
        toxic_gts = torch.stack(toxic_gts)
        toxic_gens = torch.stack(toxic_gens)
        
        anger_acc = accuracy(anger_gens, anger_gts)
        toxic_acc = accuracy(toxic_gens, toxic_gts)
        scores = {
            'anger_accuracy': anger_acc,
            'toxic_accuracy': toxic_acc
        }

        print(scores)
        return scores
    
    def load_datasets(self):
        self.train_dataset = AngerToxicDataset(self.config.TRAINING.DATA_PATH.TRAIN)
        self.dev_dataset = AngerToxicDataset(self.config.TRAINING.DATA_PATH.DEV)
        self.test_dataset = AngerToxicDataset(self.config.TRAINING.DATA_PATH.TEST)
    
    def create_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.config.TRAINING.BATCH_SIZE,
                                           collate_fn=self.train_dataloader.collate_fn)
        
        self.dev_dataloader = DataLoader(self.dev_dataset,
                                         batch_size=self.config.TRAINING.BATCH_SIZE,
                                         collate_fn=self.dev_dataloader.collate_fn)
        
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.config.TRAINING.BATCH_SIZE,
                                          collate_fn=self.test_dataloader.collate_fn)