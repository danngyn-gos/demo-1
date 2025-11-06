from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch


class SentimentDataset(Dataset):
    def __init__(self, config, df_path):
        super(SentimentDataset, self).__init__()
        self.df = pd.read_csv(df_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.PRETRAINED)

        self.text = self.df.text.tolist()
        self.empathy_score = self.df.Empathy.tolist()
        self.sentiment_score = self.df.Emotion.tolist()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'text': self.text[idx],
                'empathy': self.empathy_score[idx],
                'sentiment': self.sentiment_score[idx]}

    def collate_fn(self, batch):
        text = [item['text'] for item in batch]
        empathy = [item['empathy'] for item in batch]
        sentiment = [item['sentiment'] for item in batch]

        token_ids = self.tokenizer(text,
                                   padding=True,
                                   truncation=True,
                                   max_length=512)
        return {'text': text,
                'empathy': torch.tensor(empathy),
                'sentiment': torch.tensor(sentiment),
                'input_ids': torch.tensor(token_ids['input_ids']),
                'attention_mask': torch.tensor(token_ids['attention_mask'])}