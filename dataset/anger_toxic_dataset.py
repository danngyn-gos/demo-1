from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch


class AngerToxicDataset(Dataset):
    def __init__(self, config, df_path):
        super(AngerToxicDataset, self).__init__()
        self.df = pd.read_csv(df_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.PRETRAINED)
        self.new_special_tokens = ['<CTX>', '</CTX>', '<TARGET>', '</TARGET>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.new_special_tokens})
        self.text = self.df.text.tolist()
        self.anger = self.df.anger.tolist()
        self.toxicity = self.df.toxicity.tolist()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'text': self.text[idx],
                'anger': self.anger[idx],
                'toxic': self.toxicity[idx]}

    def collate_fn(self, batch):
        text = [item['text'] for item in batch]
        anger = [item['anger'] for item in batch]
        toxic = [item['toxic'] for item in batch]

        token_ids = self.tokenizer(text,
                                   padding=True,
                                   truncation=True,
                                   max_length=512)
        return {'text': text,
                'anger': torch.tensor(anger),
                'toxic': torch.tensor(toxic),
                'input_ids': torch.tensor(token_ids['input_ids']),
                'attention_mask': torch.tensor(token_ids['attention_mask'])}